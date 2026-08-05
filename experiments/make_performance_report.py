"""產出詳盡表現報告：從全部結果 CSV 組裝完整表格 + 解讀。

輸出：experiments/docs/performance_report.md
"""
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).parent))
from fxlib.backtest import summarize, economic_value, pt_test

RESULTS = Path(__file__).parent / "results"
DOCS = Path(__file__).parent / "docs"

LABELS = {
    "random_walk": "Random Walk", "rw_drift": "RW + drift",
    "ridge_returns": "Ridge (returns)", "ridge_returns_feat": "Ridge + 理論特徵",
    "nlinear": "NLinear", "nlinear_feat": "NLinear + 理論特徵",
    "dlinear": "DLinear", "dlinear_raw_level": "DLinear (raw level 消融)",
    "patchtst": "PatchTST-lite", "patchtst_feat": "PatchTST + 理論特徵",
    "hf_transformer": "HF Transformer (原報告模型)", "chronos_bolt": "Chronos-Bolt (零樣本)",
    "random_forest": "Random Forest", "random_forest_feat": "Random Forest + 理論特徵",
    "mlp": "MLP", "mlp_feat": "MLP + 理論特徵",
    "tuned_ridge": "Ridge (調參後)", "tuned_nlinear": "NLinear (調參後)",
    "tuned_dlinear": "DLinear (調參後)", "tuned_patchtst": "PatchTST (調參後)",
}


def lab(df, col="model"):
    df = df.copy()
    df[col] = df[col].map(lambda m: LABELS.get(m, m))
    return df


def main():
    frames = [pd.read_csv(RESULTS / "raw.csv")]
    for extra in ("raw_hf.csv", "raw_chronos.csv", "raw_tuned.csv", "raw_sklearn.csv"):
        p = RESULTS / extra
        if p.exists():
            frames.append(pd.read_csv(p))
    df = pd.concat(frames, ignore_index=True)
    meta = json.loads((RESULTS / "meta.json").read_text())
    summary = summarize(df)
    per_no = df.groupby(["model", "nation", "origin"], as_index=False).mean(numeric_only=True)

    S = []  # markdown 片段

    S.append("# 詳盡表現報告\n")
    S.append("本文件完整記錄所有實驗的表現數據。評估設定：2 個年度測試區間"
             f"（{'、'.join(f'{a}~{b}' for a, b in meta['fold_spans'])}）、"
             f"每區間 {meta['n_test_origins']} 個預測起點（stride 5）× "
             f"{meta['n_nations']} 條貨幣序列，預測 30 天路徑；"
             "神經模型 3 seeds 平均；理論特徵與所有輸入均無未來資訊洩漏。\n")

    # ---- 1. 總表 ----
    S.append("## 1. 模型總表（20 個變體，依 MASE 排序）\n")
    cols = ["model", "mase", "smape", "mae", "mse", "mase_std", "dm_stat", "dm_p",
            "pt_p_h5", "pt_p_h30"]
    S.append(lab(summary[cols]).round(4).to_markdown(index=False))
    S.append("\n- `dm_p`：與 Random Walk 的整段路徑 MAE 差異之 Diebold-Mariano p 值"
             "（dm_stat<0 表示優於 RW）\n"
             "- `pt_p_h*`：Pesaran-Timmermann 方向可預測性 p 值\n"
             "- `mase_std`：MASE 的跨起點標準差（衡量表現穩定度）\n")

    # ---- 2. 方向準確率 ----
    S.append("## 2. 方向準確率（預測 h 天後升/貶）\n")
    dcols = ["model"] + [f"diracc_h{h}" for h in (1, 5, 14, 30)]
    S.append(lab(summary[dcols].sort_values("diracc_h5", ascending=False))
             .round(4).to_markdown(index=False))
    S.append("\nRandom Walk 預測不變（方向恆為零），故其方向準確率≈0 無意義。"
             "PT 檢定已考慮方向邊際機率，比裸準確率嚴格：例如 Ridge (returns) "
             "h=30 準確率 0.593 看似最高，但因其預測方向高度偏向單邊，PT p=1.0 不顯著；"
             "Ridge + 理論特徵 0.538 反而 p<0.0001 顯著。\n")

    # ---- 3. 誤差隨視野 ----
    S.append("## 3. 各視野絕對誤差（跨貨幣×起點平均）\n")
    hor = per_no.groupby("model")[[f"abserr_h{h}" for h in (1, 5, 14, 30)]].mean()
    hor["path_mae"] = per_no.groupby("model")["mae"].mean()
    S.append(lab(hor.reset_index()).round(4).to_markdown(index=False))
    S.append("\n誤差隨視野拉長而增大（h=1 約 0.7、h=30 約 4.0），"
             "各模型差距相對於誤差本身極小（<1%），呼應點位不可預測的結論。\n")

    # ---- 4. 跨 fold ----
    S.append("## 4. 跨期穩健性（各年度區間 MASE）\n")
    fm = df.groupby(["model", "fold"])["mase"].mean().unstack()
    fm.columns = [f"{meta['fold_spans'][int(c)][0][:7]} 起一年" for c in fm.columns]
    S.append(lab(fm.reset_index()).round(3).to_markdown(index=False))
    S.append("\nfold 1（2022-03 起，美元急升年）整體 MASE 較高，"
             "但模型相對排名在兩個 fold 一致。\n")

    # ---- 5. 各貨幣 ----
    S.append("## 5. 各貨幣 MASE（代表模型）\n")
    pick = ["random_walk", "ridge_returns_feat", "patchtst", "hf_transformer"]
    nat = (per_no[per_no.model.isin(pick)]
           .groupby(["nation", "model"])["mase"].mean().unstack()[pick])
    nat.columns = [LABELS[c] for c in nat.columns]
    S.append(nat.round(3).to_markdown())
    S.append("\n北歐與亞洲貨幣（norway、sweden、japan、taiwan、south_korea）"
             "MASE 較高（波動大），歐元區與瑞郎較低；"
             "各貨幣上模型間差距均小。\n")

    # ---- 6. 經濟價值 ----
    S.append("## 6. 經濟價值：h=5 方向策略（等權組合、持有期不重疊、104 期）\n")
    ev = economic_value(df, h=5)
    S.append(lab(ev).round(3).to_markdown(index=False))
    S.append("\n- `sharpe_t`：Lo (2002) 標準誤下的 t 統計量（|t|>1.96 為 5% 顯著）\n"
             "- `sharpe_net`：扣除單邊 2bp 部位變動成本後\n"
             "- 只有 Ridge + 理論特徵顯著（t=3.26）；被動持有美元與其餘策略皆不顯著。\n")

    # ---- 7. 調參 ----
    S.append("## 7. 超參數調校結果\n")
    cc = json.loads((RESULTS / "chosen_configs.json").read_text())
    rows = []
    for k, v in cc.items():
        rows.append({"fold_模型": k, "選中配置": json.dumps(v, ensure_ascii=False)})
    S.append(pd.DataFrame(rows).to_markdown(index=False))
    S.append("\n選參只用各 fold 自身驗證區間。調參前後 MASE 對比"
             "（tuned_* vs 原設定，見第 1 節總表）變動皆 <0.1，"
             "兩個 fold 的 Ridge 都獨立選中含理論特徵版本。\n")

    # ---- 8. 敏感度 ----
    sens_p = RESULTS / "raw_sensitivity.csv"
    if sens_p.exists():
        S.append("## 8. 敏感度分析：Ridge + 理論特徵 α × 落後期全網格\n")
        sens = pd.read_csv(sens_p)
        srows = []
        for model, g in sens.groupby("model"):
            _, p5 = pt_test(g["pred_cret_h5"].values, g["true_cret_h5"].values)
            evg = economic_value(g.assign(seed=0), h=5)
            srows.append({"model": model, "mase": g.mase.mean(), "pt_p_h5": p5,
                          "sharpe_net": float(evg.loc[evg.model == model, "sharpe_net"].iloc[0])})
        sdf = pd.DataFrame(srows)
        sdf[["alpha", "n_lags"]] = sdf.model.str.extract(r"ridge_a([\d.]+)_l(\d+)").astype(float)
        S.append("PT 方向檢定 p 值（h=5）：\n")
        S.append(sdf.pivot(index="alpha", columns="n_lags", values="pt_p_h5").round(4).to_markdown())
        S.append("\n淨 Sharpe（h=5、扣 2bp）：\n")
        S.append(sdf.pivot(index="alpha", columns="n_lags", values="sharpe_net").round(2).to_markdown())
        S.append("\nMASE：\n")
        S.append(sdf.pivot(index="alpha", columns="n_lags", values="mase").round(3).to_markdown())
        S.append("\n20/20 組合方向顯著（p≤0.0042）；淨 Sharpe 全域 1.26–2.29；"
                 "MASE 變動僅 ±0.03——三個結論皆不依賴超參數。\n")

    # ---- 9. RW challenge ----
    rwc_p = RESULTS / "rwchal_dm.csv"
    if rwc_p.exists():
        S.append("## 9. 「打敗 Random Walk」專項實驗（6 年度 fold、312 起點）\n")
        rwc = pd.read_csv(rwc_p)
        S.append(rwc.round(4).to_markdown(index=False))
        S.append("\n- `*_ratio`：該模型 MAE / RW MAE（<1 為優）；`*_p`：DM p 值\n"
                 "- 最佳為 shrunk_ridge：整段路徑優 0.28%（p=0.27）、h=30 優 0.9%（p=0.096 邊緣）；"
                 "起點加密至 780 個後 h=30 p 值升至 0.68，邊緣訊號不穩健\n"
                 "- 遠期溢價 γ 在 6 個 fold 驗證集全部選 0（遠期匯率對點位無資訊）\n"
                 "- 美元因子與動能特徵（ridge_rich）未改善點位\n")

    # ---- 10. HF / Chronos 明細 ----
    S.append("## 10. 原報告模型與零樣本模型明細\n")
    hf = pd.read_csv(RESULTS / "raw_hf.csv")
    hfd = hf.groupby(["fold", "seed"])[["mase", "smape", "diracc_h30"]].mean().reset_index()
    S.append("HF TimeSeriesTransformer（無洩漏重測，各 fold × seed）：\n")
    S.append(hfd.round(3).to_markdown(index=False))
    ch = pd.read_csv(RESULTS / "raw_chronos.csv")
    chd = ch.groupby("fold")[["mase", "smape", "diracc_h30"]].mean().reset_index()
    S.append("\nChronos-Bolt（零樣本）：\n")
    S.append(chd.round(3).to_markdown(index=False))
    S.append("\nHF Transformer 的 seed 間 MASE 變異大（fold 0：4.62 vs 5.27），"
             "顯示原設定訓練不穩定；兩者點位皆劣於 RW。\n")

    # ---- 11. 結論 ----
    S.append("## 11. 表現總結\n")
    S.append("1. **點位**：20 個變體無一顯著優於 Random Walk；最佳差距僅 0.7%（p=0.21）。"
             "專項挑戰（shrinkage/遠期偏誤/美元因子、780 起點）亦然。\n"
             "2. **方向**：含理論特徵的 Ridge/RF/MLP/PatchTST 四個家族方向顯著"
             "（PT p≤0.001）；無特徵版本全不顯著（p≥0.49）。\n"
             "3. **經濟價值**：Ridge+理論特徵 5 日策略年化 8.9%、淨 Sharpe 2.16（t=3.26）；"
             "其他策略與被動基準皆不顯著。\n"
             "4. **穩健性**：上述結論跨兩個年度區間、20 組超參數、多 seed 皆成立。\n")

    DOCS.mkdir(exist_ok=True)
    (DOCS / "performance_report.md").write_text("\n".join(S), encoding="utf-8")
    print("written", DOCS / "performance_report.md")


if __name__ == "__main__":
    main()
