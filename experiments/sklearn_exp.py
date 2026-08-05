"""補測原報告的傳統 ML 模型：Random Forest 與 MLP（sklearn）。

與原報告同套件同預設精神（RF n_estimators=100；MLP 隱藏層 + Adam），
但放進無洩漏的 rolling backtest 框架，輸入與 Ridge 相同的
報酬落後期（±理論特徵）設計矩陣，目標為 h=1..30 累積 log 報酬。

用法：python experiments/sklearn_exp.py [--smoke]
"""
import argparse
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.neural_network import MLPRegressor

sys.path.insert(0, str(Path(__file__).parent))
from fxlib.data import load_panel, make_windows
from fxlib.backtest import metric_rows, mase_scales

ROOT = Path(__file__).resolve().parent.parent
L, H = 90, 30


class SkReturns:
    """與 RidgeReturns 相同的設計矩陣/還原邏輯（獨立實作以避免載入 torch），
    抽換底層 sklearn estimator。"""

    def __init__(self, name, est_fn, use_feats, n_lags=20):
        self.name = name
        self.est_fn = est_fn
        self.use_feats = use_feats
        self.n_lags = n_lags

    def _design(self, w):
        logc = np.log(w["ctx_level"])
        rets = np.diff(logc, axis=1)[:, -self.n_lags:]
        cols = [rets]
        if self.use_feats:
            f = w["ctx_feat"][:, -1, 1:]  # rate_diff, fwd_premium, rate_diff_chg5
            cols.append((f - self._f_mean) / self._f_std)
        return np.concatenate(cols, axis=1)

    def fit(self, train, val):
        f = train["ctx_feat"][:, -1, 1:]
        self._f_mean, self._f_std = f.mean(0), f.std(0) + 1e-12
        X = self._design(train)
        last = np.log(train["ctx_level"][:, -1:])
        y = np.log(train["tgt_level"]) - last
        self.model = self.est_fn()
        self.model.fit(X, y)

    def predict(self, w):
        X = self._design(w)
        last = np.log(w["ctx_level"][:, -1:])
        return np.exp(last + self.model.predict(X))


def build(seed):
    return [
        SkReturns("random_forest", lambda: RandomForestRegressor(
            n_estimators=100, random_state=seed, n_jobs=-1), use_feats=False),
        SkReturns("random_forest_feat", lambda: RandomForestRegressor(
            n_estimators=100, random_state=seed, n_jobs=-1), use_feats=True),
        SkReturns("mlp", lambda: MLPRegressor(
            hidden_layer_sizes=(64, 64), learning_rate_init=1e-3, max_iter=200,
            early_stopping=True, random_state=seed), use_feats=False),
        SkReturns("mlp_feat", lambda: MLPRegressor(
            hidden_layer_sizes=(64, 64), learning_rate_init=1e-3, max_iter=200,
            early_stopping=True, random_state=seed), use_feats=True),
    ]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--smoke", action="store_true")
    args = ap.parse_args()

    test_stride = 5
    n_test_origins = 4 if args.smoke else 52
    n_folds = 1 if args.smoke else 2
    seeds = 1 if args.smoke else 3

    names, dates, fx, feats = load_panel(ROOT / "nations")
    T, N = fx.shape[1], fx.shape[0]

    all_rows = []
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

        for seed in range(seeds):
            for model in build(seed):
                t0 = time.time()
                model.fit(train_w, val_w)
                pred = model.predict(test_w)
                rows = metric_rows(model.name, seed, test_w, pred, scales, names)
                for r in rows:
                    r["fold"] = fold
                all_rows.extend(rows)
                df_m = pd.DataFrame(rows)
                print(f"[fold {fold} seed {seed}] {model.name:20s} "
                      f"mase={df_m.mase.mean():.3f} diracc_h5={df_m.diracc_h5.mean():.3f} "
                      f"({time.time()-t0:.0f}s)", flush=True)

    out = Path(__file__).parent / "results" / (
        "raw_sklearn_smoke.csv" if args.smoke else "raw_sklearn.csv")
    pd.DataFrame(all_rows).to_csv(out, index=False)
    print("written", out)


if __name__ == "__main__":
    main()
