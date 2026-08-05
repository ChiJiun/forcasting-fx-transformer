"""專攻「打敗 Random Walk」的實驗。

三個文獻導向的策略，全部在驗證集選參、測試集只評最終配置：
  1. shrunk_ridge   : λ·Ridge+特徵 + (1−λ)·RW，λ ∈ {0.1..1.0} 驗證集選
  2. fwd_bias       : pred = s + γ·(h/30)·(FFX−s)，γ ∈ {-2..2} 驗證集選
                      （γ<0 即利用 Fama (1984) 遠期溢價偏誤）
  3. combo          : shrunk_ridge 與 fwd_bias 的等權平均
評估：6 個年度 fold（每 fold 52 起點 × 11 貨幣 = 共 312 起點），
DM 檢定分整段路徑 / h=1 / h=5 / h=30。

用法：python experiments/rw_challenge.py [--smoke]
"""
import argparse
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).parent))
from fxlib.data import load_panel, make_windows
from fxlib.models import RidgeReturns, NaiveRW
from fxlib.backtest import metric_rows, mase_scales, dm_test

ROOT = Path(__file__).resolve().parent.parent
L, H = 90, 30

LAMBDAS = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.8, 1.0]
GAMMAS = [-2.0, -1.5, -1.0, -0.5, 0.0, 0.5, 1.0, 1.5, 2.0]


class RidgeRich(RidgeReturns):
    """Ridge + 理論特徵 + 動能/反轉 + 美元因子。

    新增特徵（皆只用預測起點當下已知資訊）：
    - 自身 5/20/60 日累積 log 報酬（動能/反轉）
    - 美元因子：同一起點所有貨幣的 5/20/60 日平均累積報酬
      （Verdelhan 2018 的 dollar factor 代理）
    """

    def __init__(self, n_nations, alpha=10.0):
        super().__init__(use_feats=True, alpha=alpha)
        self.name = "ridge_rich"
        self.N = n_nations

    def _design(self, w):
        X = super()._design(w)
        logc = np.log(w["ctx_level"])
        moms = np.stack([logc[:, -1] - logc[:, -1 - k] for k in (5, 20, 60)], axis=1)
        # 視窗排序為 [origin × nation]，reshape 後對 nation 軸取平均 = 美元因子
        M = moms.reshape(-1, self.N, 3)
        dollar = np.repeat(M.mean(axis=1), self.N, axis=0)
        extra = np.concatenate([moms, dollar], axis=1)
        if not hasattr(self, "_m_mean"):
            self._m_mean, self._m_std = extra.mean(0), extra.std(0) + 1e-12
        return np.concatenate([X, (extra - self._m_mean) / self._m_std], axis=1)


def fwd_pred(w, gamma):
    """遠期匯率偏誤預測子：s_t + γ·(h/H)·(F_t − s_t)。"""
    s = w["ctx_level"][:, -1:]
    fp = w["ctx_feat"][:, -1, 1]  # fwd_premium = (F−s)/s
    hfrac = np.arange(1, H + 1)[None, :] / H
    return s * (1.0 + gamma * fp[:, None] * hfrac)


def mae(pred, w):
    return float(np.abs(pred - w["tgt_level"]).mean())


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--smoke", action="store_true")
    ap.add_argument("--stride", type=int, default=5)
    args = ap.parse_args()

    test_stride = args.stride
    n_test_origins = 4 if args.smoke else 260 // test_stride  # 維持約一年跨度
    n_folds = 1 if args.smoke else 6
    dm_lag = max(6, (H // test_stride) + 1)  # 依視窗重疊程度調整 HAC lag

    names, dates, fx, feats = load_panel(ROOT / "nations")
    T, N = fx.shape[1], fx.shape[0]

    all_rows, chosen = [], {}
    for fold in range(n_folds):
        last_origin = T - 1 - H - fold * n_test_origins * test_stride
        test_origins = np.arange(last_origin - (n_test_origins - 1) * test_stride,
                                 last_origin + 1, test_stride)
        train_cutoff = test_origins[0] - H
        val_origins = np.arange(train_cutoff - H - 250, train_cutoff - H, 5)
        train_origins = np.arange(L - 1, val_origins[0] - H, 2 if not args.smoke else 20)

        train_w = make_windows(fx, feats, train_origins, L, H)
        val_w = make_windows(fx, feats, val_origins, L, H)
        test_w = make_windows(fx, feats, test_origins, L, H)
        test_w["origin_idx"] = np.repeat(test_origins, N)
        scales = mase_scales(fx, train_cutoff)
        span = f"{str(dates[test_origins[0]])[:10]}~{str(dates[test_origins[-1]])[:10]}"

        t0 = time.time()
        ridge = RidgeReturns(use_feats=True)
        ridge.fit(train_w, val_w)
        rich = RidgeRich(n_nations=N)
        rich.fit(train_w, val_w)
        rw = NaiveRW()

        # 驗證集選 λ（兩個模型各自選）與 γ
        rv, cv, wv = ridge.predict(val_w), rich.predict(val_w), rw.predict(val_w)
        lam = min(LAMBDAS, key=lambda l: mae(l * rv + (1 - l) * wv, val_w))
        lam_r = min(LAMBDAS, key=lambda l: mae(l * cv + (1 - l) * wv, val_w))
        gam = min(GAMMAS, key=lambda g: mae(fwd_pred(val_w, g), val_w))
        chosen[f"fold{fold}"] = {"lambda": lam, "lambda_rich": lam_r,
                                 "gamma": gam, "span": span}

        # 測試集：只評最終配置
        rt, ct, wt = ridge.predict(test_w), rich.predict(test_w), rw.predict(test_w)
        preds = {
            "random_walk": wt,
            "ridge_feat_6f": rt,
            "shrunk_ridge": lam * rt + (1 - lam) * wt,
            "ridge_rich": ct,
            "shrunk_rich": lam_r * ct + (1 - lam_r) * wt,
            "fwd_bias": fwd_pred(test_w, gam),
        }
        preds["combo"] = 0.5 * preds["shrunk_rich"] + 0.5 * preds["shrunk_ridge"]

        for name, pred in preds.items():
            rows = metric_rows(name, 0, test_w, pred, scales, names)
            for r in rows:
                r["fold"] = fold
            all_rows.extend(rows)
        print(f"[fold {fold}] {span} lambda={lam} gamma={gam} "
              f"({time.time()-t0:.0f}s)", flush=True)

    df = pd.DataFrame(all_rows)
    out = Path(__file__).parent / "results"
    df.to_csv(out / ("raw_rwchal_smoke.csv" if args.smoke else "raw_rwchal.csv"),
              index=False)

    # ---- DM 檢定：整段路徑與各視野 ----
    per_no = df.groupby(["model", "nation", "origin"], as_index=False).mean(numeric_only=True)
    base = per_no[per_no.model == "random_walk"]
    print("\n=== vs Random Walk（跨全部 fold 合併）===")
    results = []
    for model in preds:
        if model == "random_walk":
            continue
        g = per_no[per_no.model == model]
        row = {"model": model}
        for col, tag in [("mae", "path"), ("abserr_h1", "h1"),
                         ("abserr_h5", "h5"), ("abserr_h30", "h30")]:
            a = g.groupby("origin")[col].mean().sort_index()
            b = base.groupby("origin")[col].mean().sort_index()
            dm, p = dm_test(a.values, b.values, max_lag=dm_lag)
            row[f"{tag}_ratio"] = a.mean() / b.mean()
            row[f"{tag}_dm"] = dm
            row[f"{tag}_p"] = p
        results.append(row)
    res = pd.DataFrame(results)
    res.to_csv(out / "rwchal_dm.csv", index=False)
    with pd.option_context("display.width", 250):
        print(res.round(4).to_string(index=False))
    print("\nchosen:", chosen)


if __name__ == "__main__":
    main()
