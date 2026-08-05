"""跑完整實驗矩陣：rolling-origin 回測 × 模型變體 × 多 seed。

用法：
    python experiments/run_experiments.py            # 完整
    python experiments/run_experiments.py --smoke    # 快速冒煙測試
"""
import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).parent))
from fxlib.data import load_panel, make_windows
from fxlib.models import build_models, TorchForecaster
from fxlib.backtest import metric_rows, mase_scales, summarize

ROOT = Path(__file__).resolve().parent.parent


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--smoke", action="store_true")
    ap.add_argument("--seeds", type=int, default=3)
    ap.add_argument("--out", default=str(Path(__file__).parent / "results"))
    args = ap.parse_args()

    L, H = 90, 30
    test_stride = 5
    n_test_origins = 8 if args.smoke else 52
    n_folds = 1 if args.smoke else 2
    seeds = 1 if args.smoke else args.seeds

    names, dates, fx, feats = load_panel(ROOT / "nations")
    T = fx.shape[1]

    all_rows = []
    fold_spans = []
    for fold in range(n_folds):
        # ---- 每個 fold 一個年度測試區間，各自重訓（嚴格依時間，含 embargo）----
        last_origin = T - 1 - H - fold * n_test_origins * test_stride
        test_origins = np.arange(last_origin - (n_test_origins - 1) * test_stride,
                                 last_origin + 1, test_stride)
        train_cutoff = test_origins[0] - H  # 訓練/驗證視窗的 target 不可踩進測試區
        val_origins = np.arange(train_cutoff - H - 250, train_cutoff - H, 5)
        train_origins = np.arange(L - 1, val_origins[0] - H, 2 if not args.smoke else 20)

        span = [str(dates[test_origins[0]])[:10], str(dates[test_origins[-1]])[:10]]
        fold_spans.append(span)
        print(f"fold {fold}: train origins {len(train_origins)}  val {len(val_origins)}  "
              f"test {len(test_origins)}  span {span[0]} ~ {span[1]}", flush=True)

        train_w = make_windows(fx, feats, train_origins, L, H)
        val_w = make_windows(fx, feats, val_origins, L, H)
        test_w = make_windows(fx, feats, test_origins, L, H)
        test_w["origin_idx"] = np.repeat(test_origins, fx.shape[0])

        scales = mase_scales(fx, train_cutoff)

        for seed in range(seeds):
            for model in build_models(seed=seed):
                if seed > 0 and not isinstance(model, TorchForecaster):
                    continue  # 確定性模型只跑一次
                t0 = time.time()
                model.fit(train_w, val_w)
                pred = model.predict(test_w)
                rows = metric_rows(model.name, seed, test_w, pred, scales, names)
                for r in rows:
                    r["fold"] = fold
                all_rows.extend(rows)
                df_m = pd.DataFrame(rows)
                print(f"[fold {fold} seed {seed}] {model.name:22s} "
                      f"mase={df_m.mase.mean():.3f} smape={df_m.smape.mean():.4f} "
                      f"diracc_h30={df_m.diracc_h30.mean():.3f} "
                      f"({time.time()-t0:.0f}s)", flush=True)

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    df = pd.DataFrame(all_rows)
    df.to_csv(out / ("raw_smoke.csv" if args.smoke else "raw.csv"), index=False)

    summary = summarize(df)
    summary.to_csv(out / ("summary_smoke.csv" if args.smoke else "summary.csv"), index=False)
    with pd.option_context("display.width", 200):
        print(summary.to_string(index=False))

    meta = {
        "context_len": L, "horizon": H, "test_stride": test_stride,
        "n_test_origins": int(n_test_origins), "n_folds": n_folds, "seeds": seeds,
        "fold_spans": fold_spans, "n_nations": int(fx.shape[0]),
        "nations": names,
        "n_train_windows": int(train_w["ctx_level"].shape[0]),
    }
    (out / "meta.json").write_text(json.dumps(meta, indent=2))


if __name__ == "__main__":
    main()
