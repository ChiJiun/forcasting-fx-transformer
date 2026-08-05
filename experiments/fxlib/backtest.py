"""Rolling-origin 回測與評估指標。

- 多個預測起點（stride 抽樣），每起點預測 H 天路徑
- MASE / sMAPE / MAE / MSE / 方向準確率（h=1,5,14,30）
- Diebold-Mariano 檢定（HAC 變異數，起點重疊時仍有效）
"""
import numpy as np
import pandas as pd
from scipy import stats


def metric_rows(model_name, seed, w, pred, mase_scale, names, horizons=(1, 5, 14, 30)):
    """每個 (nation, origin) 一列的指標。w 需含 origin_idx 欄位。"""
    tgt = w["tgt_level"]
    last = w["ctx_level"][:, -1]
    err = pred - tgt
    rows = []
    for i in range(tgt.shape[0]):
        n = w["nation_id"][i]
        row = {
            "model": model_name,
            "seed": seed,
            "nation": names[n],
            "origin": int(w["origin_idx"][i]),
            "mae": float(np.abs(err[i]).mean()),
            "mse": float((err[i] ** 2).mean()),
            "smape": float(np.mean(2 * np.abs(err[i]) / (np.abs(pred[i]) + np.abs(tgt[i])))),
            "mase": float(np.abs(err[i]).mean() / mase_scale[n]),
        }
        for h in horizons:
            true_dir = np.sign(tgt[i, h - 1] - last[i])
            pred_dir = np.sign(pred[i, h - 1] - last[i])
            row[f"diracc_h{h}"] = float(true_dir == pred_dir)
            row[f"abserr_h{h}"] = float(np.abs(err[i, h - 1]))
            # 帶正負號的累積 log return（經濟價值回測與 PT 檢定用）
            row[f"pred_cret_h{h}"] = float(np.log(pred[i, h - 1] / last[i]))
            row[f"true_cret_h{h}"] = float(np.log(tgt[i, h - 1] / last[i]))
        rows.append(row)
    return rows


def mase_scales(fx, train_end):
    """MASE 分母：訓練期 naive 一步預測的平均絕對誤差（每國一個）。"""
    return np.abs(np.diff(fx[:, :train_end], axis=1)).mean(axis=1)


def dm_test(loss_a, loss_b, max_lag=6):
    """Diebold-Mariano：loss_a/loss_b 為依時間排序的每起點損失序列。

    回傳 (dm_stat, p_value)。dm<0 表示 A 優於 B。
    HAC (Bartlett kernel) 變異數處理重疊預測造成的自相關。
    """
    d = np.asarray(loss_a, dtype=float) - np.asarray(loss_b, dtype=float)
    T = len(d)
    max_lag = min(max_lag, max(1, T // 4))  # 樣本太少時縮小 lag，避免 HAC 退化
    dbar = d.mean()
    dc = d - dbar
    gamma0 = (dc ** 2).mean()
    var = gamma0
    for k in range(1, min(max_lag, T - 1) + 1):
        gk = (dc[k:] * dc[:-k]).mean()
        var += 2 * (1 - k / (max_lag + 1)) * gk
    if var <= 0:
        return 0.0, 1.0
    dm = dbar / np.sqrt(var / T)
    # Harvey-Leybourne-Newbold 小樣本修正
    h = max_lag
    correction = np.sqrt(max(1e-12, (T + 1 - 2 * h + h * (h - 1) / T) / T))
    dm *= correction
    p = 2 * stats.t.sf(np.abs(dm), df=T - 1)
    return float(dm), float(p)


def pt_test(pred_ret, true_ret):
    """Pesaran-Timmermann (1992) 方向可預測性檢定。

    H0：預測方向與實際方向獨立。回傳 (stat, p)，stat>0 且 p 小代表
    方向預測能力顯著優於隨機。
    """
    x = (np.asarray(pred_ret) > 0).astype(float)
    y = (np.asarray(true_ret) > 0).astype(float)
    n = len(x)
    p_hat = (x == y).mean()
    py, px = y.mean(), x.mean()
    p_star = py * px + (1 - py) * (1 - px)
    v_hat = p_star * (1 - p_star) / n
    v_star = ((2 * py - 1) ** 2 * px * (1 - px) / n
              + (2 * px - 1) ** 2 * py * (1 - py) / n
              + 4 * py * px * (1 - py) * (1 - px) / n ** 2)
    denom = v_hat - v_star
    if denom <= 0:
        return 0.0, 1.0
    stat = (p_hat - p_star) / np.sqrt(denom)
    return float(stat), float(stats.norm.sf(stat))


def economic_value(df, h=5, periods_per_year=252, cost_oneway=0.0002):
    """簡單方向策略的經濟價值：每個起點依 h 天預測方向持有 h 天。

    stride=5、h=5 時各持有期不重疊。報酬為等權跨幣別組合的
    log return（FX 為每美元兌換的外幣數，pred>0 = 看多美元）。
    附 Lo (2002) Sharpe 標準誤的 t 統計量，以及扣除單邊
    cost_oneway（預設 2bp）交易成本後的淨 Sharpe。
    """
    per_no = (df.groupby(["model", "nation", "origin"], as_index=False)
                .mean(numeric_only=True))
    scale = periods_per_year / h

    def stats_row(model, port):
        mu, sd = port.mean(), port.std()
        n = len(port)
        sr_p = mu / sd if sd > 0 else np.nan  # 每期 Sharpe
        se = np.sqrt((1 + 0.5 * sr_p ** 2) / n) if n > 1 else np.nan
        return {"model": model, "ann_ret": mu * scale,
                "ann_vol": sd * np.sqrt(scale),
                "sharpe": sr_p * np.sqrt(scale) if sd > 0 else np.nan,
                "sharpe_t": sr_p / se if se and se > 0 else np.nan}

    out = []
    for model, g in per_no.groupby("model"):
        g = g.sort_values(["nation", "origin"])
        pos = np.sign(g[f"pred_cret_h{h}"])
        # 交易成本：部位變動 |Δpos| 單位 × 單邊成本（各幣別獨立計算）
        prev = pos.groupby(g["nation"].values).shift(1).fillna(0.0)
        cost = cost_oneway * (pos - prev).abs()
        gross = pos * g[f"true_cret_h{h}"]
        port_g = gross.groupby(g["origin"]).mean().sort_index()
        port_n = (gross - cost).groupby(g["origin"]).mean().sort_index()
        row = stats_row(model, port_g)
        row["sharpe_net"] = stats_row(model, port_n)["sharpe"]
        out.append(row)
    g0 = per_no[per_no.model == per_no.model.iloc[0]]
    port = g0[f"true_cret_h{h}"].groupby(g0["origin"]).mean().sort_index()
    row = stats_row("long_usd_passive", port)
    row["sharpe_net"] = row["sharpe"]  # 被動部位不換手
    out.append(row)
    return pd.DataFrame(out).sort_values("sharpe", ascending=False).reset_index(drop=True)


def summarize(df, baseline="random_walk"):
    """彙整成模型層級摘要表，含 DM 檢定 vs baseline。"""
    # 對神經模型多 seed 先取平均（同 nation/origin）
    per_no = (df.groupby(["model", "nation", "origin"], as_index=False)
                .mean(numeric_only=True).drop(columns="seed"))
    base = (per_no[per_no.model == baseline]
            .groupby("origin")["mae"].mean().sort_index())

    out = []
    for model, g in per_no.groupby("model"):
        agg = g.mean(numeric_only=True)
        row = {"model": model,
               "mase": agg["mase"], "smape": agg["smape"],
               "mae": agg["mae"], "mse": agg["mse"]}
        for h in (1, 5, 14, 30):
            row[f"diracc_h{h}"] = agg[f"diracc_h{h}"]
        # MASE 的起點間標準差（跨國平均後）
        by_origin = g.groupby("origin")[["mase", "mae"]].mean().sort_index()
        row["mase_std"] = by_origin["mase"].std()
        if model != baseline:
            dm, p = dm_test(by_origin["mae"].values, base.values)
            row["dm_stat"], row["dm_p"] = dm, p
        else:
            row["dm_stat"], row["dm_p"] = np.nan, np.nan
        for h in (5, 30):
            _, p = pt_test(g[f"pred_cret_h{h}"].values, g[f"true_cret_h{h}"].values)
            row[f"pt_p_h{h}"] = p
        out.append(row)
    return pd.DataFrame(out).sort_values("mase").reset_index(drop=True)
