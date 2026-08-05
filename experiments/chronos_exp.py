"""Chronos-Bolt 零樣本 baseline：不做任何訓練，直接預測同一組測試起點。

注意：Chronos 為預訓練基礎模型，其語料可能與部分金融序列重疊，
結果應標註為 zero-shot 參考基準而非嚴格意義的 out-of-sample。
"""
import argparse
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch

sys.path.insert(0, str(Path(__file__).parent))
from fxlib.data import load_panel, make_windows
from fxlib.backtest import metric_rows, mase_scales

ROOT = Path(__file__).resolve().parent.parent
H = 30
CTX = 512  # chronos-bolt 的最大 context


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--smoke", action="store_true")
    ap.add_argument("--model", default="amazon/chronos-bolt-small")
    args = ap.parse_args()

    from chronos import BaseChronosPipeline
    pipe = BaseChronosPipeline.from_pretrained(
        args.model, device_map="cpu", torch_dtype=torch.float32)

    test_stride = 5
    n_test_origins = 4 if args.smoke else 52
    n_folds = 1 if args.smoke else 2

    names, dates, fx, feats = load_panel(ROOT / "nations")
    T, N = fx.shape[1], fx.shape[0]

    all_rows = []
    for fold in range(n_folds):
        last_origin = T - 1 - H - fold * n_test_origins * test_stride
        test_origins = np.arange(last_origin - (n_test_origins - 1) * test_stride,
                                 last_origin + 1, test_stride)
        train_cutoff = test_origins[0] - H
        test_w = make_windows(fx, feats, test_origins, CTX, H)
        test_w["origin_idx"] = np.repeat(test_origins, N)
        scales = mase_scales(fx, train_cutoff)

        t0 = time.time()
        ctx = torch.tensor(test_w["ctx_level"], dtype=torch.float32)
        preds = []
        for i in range(0, ctx.shape[0], 64):
            q, _ = pipe.predict_quantiles(
                ctx[i:i + 64], prediction_length=H, quantile_levels=[0.5])
            preds.append(q[:, :, 0].numpy())
        pred = np.vstack(preds)
        rows = metric_rows("chronos_bolt", 0, test_w, pred, scales, names)
        for r in rows:
            r["fold"] = fold
        all_rows.extend(rows)
        df_m = pd.DataFrame(rows)
        print(f"[fold {fold}] chronos_bolt mase={df_m.mase.mean():.3f} "
              f"smape={df_m.smape.mean():.4f} diracc_h30={df_m.diracc_h30.mean():.3f} "
              f"({time.time()-t0:.0f}s)", flush=True)

    out_path = Path(__file__).parent / "results" / (
        "raw_chronos_smoke.csv" if args.smoke else "raw_chronos.csv")
    pd.DataFrame(all_rows).to_csv(out_path, index=False)
    print("written", out_path)


if __name__ == "__main__":
    main()
