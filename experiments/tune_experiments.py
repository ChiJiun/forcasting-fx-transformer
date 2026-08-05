"""超參數調校：在驗證區間選參，測試集只評估選出的配置。

流程（每個 fold 獨立）：
  1. 對每組候選超參數：用訓練視窗訓練 → 在驗證起點上評估 MAE
  2. 選驗證 MAE 最低的配置
  3. 只把選出的配置拿到測試起點評估（tuned_* 模型名寫進 raw_tuned.csv）
另輸出：Ridge+理論特徵在整個 α×lags 網格上的測試表現（敏感度分析用，
非選參依據）。

用法：python experiments/tune_experiments.py [--smoke]
"""
import argparse
import itertools
import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).parent))
from fxlib.data import load_panel, make_windows
from fxlib.models import (RidgeReturns, TorchForecaster,
                          _NLinearNet, _DLinearNet, _PatchTSTNet)
from fxlib.backtest import metric_rows, mase_scales

ROOT = Path(__file__).resolve().parent.parent
H = 30


def eval_mae(model, w):
    pred = model.predict(w)
    return float(np.abs(pred - w["tgt_level"]).mean()), pred


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--smoke", action="store_true")
    args = ap.parse_args()

    test_stride = 5
    n_test_origins = 4 if args.smoke else 52
    n_folds = 1 if args.smoke else 2
    final_seeds = 1 if args.smoke else 3

    names, dates, fx, feats = load_panel(ROOT / "nations")
    T, N = fx.shape[1], fx.shape[0]

    # ---- 候選網格 ----
    ridge_grid = [{"alpha": a, "n_lags": l, "use_feats": f}
                  for a in (0.1, 1.0, 10.0, 100.0, 1000.0)
                  for l in (5, 10, 20, 40)
                  for f in (False, True)]
    nlinear_grid = [{"L": L, "lr": lr}
                    for L in (30, 60, 90, 180) for lr in (1e-3, 3e-4)]
    dlinear_grid = [{"L": L, "lr": lr, "kernel": k}
                    for L in (60, 90, 180) for lr in (1e-3,) for k in (13, 25, 51)]
    patch_grid = [{"L": 90, "lr": lr, "d": d, "layers": ly, "patch": p}
                  for lr in (1e-3, 3e-4) for d in (32, 64) for ly in (2,)
                  for p in (10,)] + [
                  {"L": 90, "lr": 1e-3, "d": 64, "layers": 3, "patch": 10},
                  {"L": 90, "lr": 1e-3, "d": 64, "layers": 2, "patch": 15},
                  {"L": 180, "lr": 1e-3, "d": 64, "layers": 2, "patch": 15}]
    if args.smoke:
        ridge_grid = ridge_grid[:4]
        nlinear_grid = nlinear_grid[:2]
        dlinear_grid = dlinear_grid[:2]
        patch_grid = patch_grid[:1]

    all_rows, sensitivity_rows, chosen = [], [], {}
    for fold in range(n_folds):
        last_origin = T - 1 - H - fold * n_test_origins * test_stride
        test_origins = np.arange(last_origin - (n_test_origins - 1) * test_stride,
                                 last_origin + 1, test_stride)
        train_cutoff = test_origins[0] - H
        val_origins = np.arange(train_cutoff - H - 250, train_cutoff - H, 5)
        scales = mase_scales(fx, train_cutoff)

        # 各 context 長度的視窗快取
        win_cache = {}

        def wins(L):
            if L not in win_cache:
                tr = np.arange(L - 1, val_origins[0] - H, 2 if not args.smoke else 20)
                win_cache[L] = (
                    make_windows(fx, feats, tr, L, H),
                    make_windows(fx, feats, val_origins, L, H),
                    make_windows(fx, feats, test_origins, L, H),
                )
            return win_cache[L]

        def add_test_rows(name, model, L, seed=0):
            _, _, test_w = wins(L)
            test_w = dict(test_w)
            test_w["origin_idx"] = np.repeat(test_origins, N)
            _, pred = eval_mae(model, test_w)
            rows = metric_rows(name, seed, test_w, pred, scales, names)
            for r in rows:
                r["fold"] = fold
            all_rows.extend(rows)
            return pd.DataFrame(rows)

        # ---------------- Ridge：全網格 ----------------
        t0 = time.time()
        best = (np.inf, None)
        for cfg in ridge_grid:
            train_w, val_w, test_w = wins(90)
            m = RidgeReturns(use_feats=cfg["use_feats"], n_lags=cfg["n_lags"],
                             alpha=cfg["alpha"])
            m.fit(train_w, val_w)
            vmae, _ = eval_mae(m, val_w)
            if vmae < best[0]:
                best = (vmae, cfg, m)
            # 敏感度分析：帶特徵版的整個網格都記錄測試表現（非選參用）
            if cfg["use_feats"]:
                tw = dict(test_w)
                tw["origin_idx"] = np.repeat(test_origins, N)
                _, pred = eval_mae(m, tw)
                rows = metric_rows(f"ridge_a{cfg['alpha']}_l{cfg['n_lags']}", 0,
                                   tw, pred, scales, names)
                for r in rows:
                    r["fold"] = fold
                sensitivity_rows.extend(rows)
        chosen[f"fold{fold}_ridge"] = best[1]
        df_m = add_test_rows("tuned_ridge", best[2], 90)
        print(f"[fold {fold}] tuned_ridge {best[1]} val_mae={best[0]:.3f} "
              f"test mase={df_m.mase.mean():.3f} ({time.time()-t0:.0f}s)", flush=True)

        # ---------------- NLinear / DLinear / PatchTST ----------------
        def tune_torch(tag, grid, net_builder):
            t0 = time.time()
            best = (np.inf, None, None)
            for cfg in grid:
                train_w, val_w, _ = wins(cfg["L"])
                m = TorchForecaster(f"{tag}_probe", net_builder(cfg),
                                    seed=0, lr=cfg["lr"])
                m.fit(train_w, val_w)
                vmae, _ = eval_mae(m, val_w)
                if vmae < best[0]:
                    best = (vmae, cfg, m)
            cfg = best[1]
            chosen[f"fold{fold}_{tag}"] = cfg
            # 最終配置以多 seeds 重訓（seed 0 沿用已訓練的）
            for seed in range(final_seeds):
                if seed == 0:
                    m = best[2]
                else:
                    train_w, val_w, _ = wins(cfg["L"])
                    m = TorchForecaster(f"{tag}_probe", net_builder(cfg),
                                        seed=seed, lr=cfg["lr"])
                    m.fit(train_w, val_w)
                df_m = add_test_rows(f"tuned_{tag}", m, cfg["L"], seed)
            print(f"[fold {fold}] tuned_{tag} {cfg} val_mae={best[0]:.3f} "
                  f"test mase={df_m.mase.mean():.3f} ({time.time()-t0:.0f}s)", flush=True)

        tune_torch("nlinear", nlinear_grid,
                   lambda cfg: (lambda L, H_: _NLinearNet(L, H_)))
        tune_torch("dlinear", dlinear_grid,
                   lambda cfg: (lambda L, H_, k=cfg["kernel"]: _DLinearNet(L, H_, kernel=k)))
        tune_torch("patchtst", patch_grid,
                   lambda cfg: (lambda L, H_, c=cfg: _PatchTSTNet(
                       L, H_, patch_len=c["patch"], d_model=c["d"], n_layers=c["layers"])))

    out = Path(__file__).parent / "results"
    suffix = "_smoke" if args.smoke else ""
    pd.DataFrame(all_rows).to_csv(out / f"raw_tuned{suffix}.csv", index=False)
    pd.DataFrame(sensitivity_rows).to_csv(out / f"raw_sensitivity{suffix}.csv", index=False)
    (out / f"chosen_configs{suffix}.json").write_text(
        json.dumps(chosen, indent=2, default=str))
    print("chosen:", json.dumps(chosen, default=str))


if __name__ == "__main__":
    main()
