"""原報告的 HF TimeSeriesTransformer，以無洩漏設定接進 rolling backtest。

與原 transformer.py 的差異：
- 不把未來 FFX/利率餵給模型（num_dynamic_real_features=0；時間特徵只有日曆與 age，
  這些在未來視窗是真正已知的）
- 直接用 numpy 視窗驅動模型（繞過 gluonts），與其他模型共用同一組
  訓練/驗證/測試起點與 embargo
- early stopping + 依驗證 loss 選 checkpoint

用法：python experiments/hf_transformer_exp.py [--smoke] [--seeds 2]
"""
import argparse
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from transformers import TimeSeriesTransformerConfig, TimeSeriesTransformerForPrediction

sys.path.insert(0, str(Path(__file__).parent))
from fxlib.data import load_panel, make_windows
from fxlib.backtest import metric_rows, mase_scales

ROOT = Path(__file__).resolve().parent.parent

LAGS = [1, 2, 3, 4, 5, 6, 7, 11, 12, 13, 23, 24, 25, 35, 36, 37]  # 同原報告
CONTEXT = 90
H = 30
PAST_LEN = CONTEXT + max(LAGS)  # 模型需要 context + max lag 的歷史


def time_feature_matrix(dates):
    """[T, 4]：day-of-week / day-of-month / day-of-year（gluonts 慣例縮放）+ log age。"""
    idx = pd.DatetimeIndex(dates)
    dow = idx.dayofweek.values / 6.0 - 0.5
    dom = (idx.day.values - 1) / 30.0 - 0.5
    doy = (idx.dayofyear.values - 1) / 365.0 - 0.5
    age = np.log10(2.0 + np.arange(len(idx)))
    return np.stack([dow, dom, doy, age], axis=-1).astype(np.float32)


def build_tensors(w, tf_all, origins, n_nations):
    """把 make_windows 的輸出轉成 HF 模型的輸入張量。"""
    B = w["ctx_level"].shape[0]
    past_values = torch.tensor(w["ctx_level"], dtype=torch.float32)
    future_values = torch.tensor(w["tgt_level"], dtype=torch.float32)
    static_cat = torch.tensor(w["nation_id"], dtype=torch.long).unsqueeze(-1)
    # 每個視窗的時間特徵（依 origin 對齊；同一 origin 下 11 國相同）
    past_tf, future_tf = [], []
    for t in origins:
        p = tf_all[t - PAST_LEN + 1: t + 1]
        f = tf_all[t + 1: t + 1 + H]
        past_tf.extend([p] * n_nations)
        future_tf.extend([f] * n_nations)
    return {
        "past_values": past_values,
        "past_time_features": torch.tensor(np.asarray(past_tf)),
        "past_observed_mask": torch.ones_like(past_values),
        "future_values": future_values,
        "future_time_features": torch.tensor(np.asarray(future_tf)),
        "static_categorical_features": static_cat,
    }


def batches(tensors, batch_size, shuffle=True):
    B = tensors["past_values"].shape[0]
    order = torch.randperm(B) if shuffle else torch.arange(B)
    for i in range(0, B, batch_size):
        idx = order[i:i + batch_size]
        yield {k: v[idx] for k, v in tensors.items()}


def val_loss(model, tensors, batch_size=256):
    model.eval()
    total, n = 0.0, 0
    with torch.no_grad():
        for b in batches(tensors, batch_size, shuffle=False):
            out = model(**b)
            total += out.loss.item() * b["past_values"].shape[0]
            n += b["past_values"].shape[0]
    return total / n


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--smoke", action="store_true")
    ap.add_argument("--seeds", type=int, default=2)
    args = ap.parse_args()

    test_stride = 5
    n_test_origins = 4 if args.smoke else 52
    n_folds = 1 if args.smoke else 2
    seeds = 1 if args.smoke else args.seeds
    max_epochs = 2 if args.smoke else 25
    patience = 4

    names, dates, fx, feats = load_panel(ROOT / "nations")
    T = fx.shape[1]
    N = fx.shape[0]
    tf_all = time_feature_matrix(dates)

    all_rows = []
    for fold in range(n_folds):
        last_origin = T - 1 - H - fold * n_test_origins * test_stride
        test_origins = np.arange(last_origin - (n_test_origins - 1) * test_stride,
                                 last_origin + 1, test_stride)
        train_cutoff = test_origins[0] - H
        val_origins = np.arange(train_cutoff - H - 250, train_cutoff - H, 5)
        train_origins = np.arange(PAST_LEN - 1, val_origins[0] - H,
                                  3 if not args.smoke else 40)
        print(f"fold {fold}: train {len(train_origins)} val {len(val_origins)} "
              f"test {len(test_origins)} origins", flush=True)

        train_w = make_windows(fx, feats, train_origins, PAST_LEN, H)
        val_w = make_windows(fx, feats, val_origins, PAST_LEN, H)
        test_w = make_windows(fx, feats, test_origins, PAST_LEN, H)
        test_w["origin_idx"] = np.repeat(test_origins, N)
        scales = mase_scales(fx, train_cutoff)

        train_t = build_tensors(train_w, tf_all, train_origins, N)
        val_t = build_tensors(val_w, tf_all, val_origins, N)
        test_t = build_tensors(test_w, tf_all, test_origins, N)
        test_t.pop("future_values")

        for seed in range(seeds):
            torch.manual_seed(seed)
            np.random.seed(seed)
            config = TimeSeriesTransformerConfig(
                prediction_length=H,
                context_length=CONTEXT,
                lags_sequence=LAGS,
                num_time_features=tf_all.shape[1],
                num_static_categorical_features=1,
                num_dynamic_real_features=0,  # 關鍵：不用任何未來不可知的共變數
                cardinality=[N],
                embedding_dimension=[1],
                encoder_layers=4,
                decoder_layers=4,
                d_model=32,
            )
            model = TimeSeriesTransformerForPrediction(config)
            opt = torch.optim.AdamW(model.parameters(), lr=4e-4,
                                    betas=(0.9, 0.95), weight_decay=1e-1)

            best_val, best_state, bad = np.inf, None, 0
            for epoch in range(max_epochs):
                t0 = time.time()
                model.train()
                tl, nb = 0.0, 0
                for b in batches(train_t, 128):
                    opt.zero_grad()
                    out = model(**b)
                    out.loss.backward()
                    opt.step()
                    tl += out.loss.item()
                    nb += 1
                vl = val_loss(model, val_t)
                print(f"[fold {fold} seed {seed}] epoch {epoch} "
                      f"train {tl/nb:.4f} val {vl:.4f} ({time.time()-t0:.0f}s)", flush=True)
                if vl < best_val - 1e-4:
                    best_val, bad = vl, 0
                    best_state = {k: v.clone() for k, v in model.state_dict().items()}
                else:
                    bad += 1
                    if bad >= patience:
                        break
            if best_state is not None:
                model.load_state_dict(best_state)

            model.eval()
            preds = []
            with torch.no_grad():
                for b in batches(test_t, 64, shuffle=False):
                    out = model.generate(**b)
                    preds.append(np.median(out.sequences.numpy(), axis=1))
            pred = np.vstack(preds)
            rows = metric_rows("hf_transformer", seed, test_w, pred, scales, names)
            for r in rows:
                r["fold"] = fold
            all_rows.extend(rows)
            df_m = pd.DataFrame(rows)
            print(f"[fold {fold} seed {seed}] hf_transformer mase={df_m.mase.mean():.3f} "
                  f"diracc_h30={df_m.diracc_h30.mean():.3f}", flush=True)

    out_path = Path(__file__).parent / "results" / (
        "raw_hf_smoke.csv" if args.smoke else "raw_hf.csv")
    pd.DataFrame(all_rows).to_csv(out_path, index=False)
    print("written", out_path)


if __name__ == "__main__":
    main()
