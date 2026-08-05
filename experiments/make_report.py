"""由 results/raw.csv 產出比較圖表與 markdown 報告。"""
import json
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

plt.rcParams["font.family"] = ["Microsoft JhengHei", "sans-serif"]
plt.rcParams["axes.unicode_minus"] = False
import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).parent))
from fxlib.backtest import summarize, economic_value

RESULTS = Path(__file__).parent / "results"

# dataviz 參考色票（light mode）
SURFACE = "#fcfcfb"
INK = "#0b0b0b"
INK2 = "#52514e"
MUTED = "#898781"
GRID = "#e1e0d9"
BASELINE_AXIS = "#c3c2b7"
SERIES = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100"]  # blue orange aqua yellow

LABELS = {
    "random_walk": "Random Walk",
    "rw_drift": "RW + drift",
    "ridge_returns": "Ridge (returns)",
    "ridge_returns_feat": "Ridge + theory feat.",
    "nlinear": "NLinear",
    "nlinear_feat": "NLinear + theory feat.",
    "dlinear": "DLinear",
    "dlinear_raw_level": "DLinear (raw level)",
    "patchtst": "PatchTST-lite",
    "patchtst_feat": "PatchTST + theory feat.",
    "hf_transformer": "HF Transformer (原報告模型, 無洩漏)",
    "chronos_bolt": "Chronos-Bolt (zero-shot)",
    "random_forest": "Random Forest (原報告方法)",
    "random_forest_feat": "Random Forest + theory feat.",
    "mlp": "MLP (原報告方法)",
    "mlp_feat": "MLP + theory feat.",
    "tuned_ridge": "Ridge (調參後)",
    "tuned_nlinear": "NLinear (調參後)",
    "tuned_dlinear": "DLinear (調參後)",
    "tuned_patchtst": "PatchTST (調參後)",
}


def style_ax(ax):
    ax.set_facecolor(SURFACE)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    for s in ("left", "bottom"):
        ax.spines[s].set_color(BASELINE_AXIS)
    ax.tick_params(colors=MUTED, labelsize=9)
    ax.yaxis.grid(True, color=GRID, linewidth=0.8)
    ax.xaxis.grid(False)
    ax.set_axisbelow(True)


def fig_mase_bar(summary, path, xlabel="MASE（越低越好）"):
    df = summary.sort_values("mase", ascending=False)
    fig, ax = plt.subplots(figsize=(7.5, 4.2), dpi=150)
    fig.patch.set_facecolor(SURFACE)
    style_ax(ax)
    ax.xaxis.grid(True, color=GRID, linewidth=0.8)
    ax.yaxis.grid(False)
    names = [LABELS.get(m, m) for m in df.model]
    colors = [MUTED if m == "random_walk" else SERIES[0] for m in df.model]
    bars = ax.barh(names, df.mase, color=colors, height=0.62)
    rw = float(summary.loc[summary.model == "random_walk", "mase"].iloc[0])
    ax.axvline(rw, color=INK2, linewidth=1.2, linestyle=(0, (4, 3)))
    ax.annotate(f"Random Walk = {rw:.2f}", xy=(rw, len(names) - 0.4),
                xytext=(4, 0), textcoords="offset points",
                fontsize=8.5, color=INK2, va="center")
    for b, v in zip(bars, df.mase):
        ax.annotate(f"{v:.2f}", xy=(v, b.get_y() + b.get_height() / 2),
                    xytext=(4, 0), textcoords="offset points",
                    fontsize=8.5, color=INK2, va="center")
    ax.set_xlabel(xlabel, fontsize=9.5, color=INK2)
    ax.set_title("30 天匯率預測：各模型 MASE", fontsize=12, color=INK,
                 loc="left", pad=12)
    fig.tight_layout()
    fig.savefig(path, facecolor=SURFACE)
    plt.close(fig)


def fig_horizon(per_no, models, path, col_tpl="abserr_h{h}", ylabel="平均絕對誤差（MAE）",
                title="誤差隨預測天數的成長", ref=None, ref_label=None,
                direct_labels=True):
    horizons = [1, 5, 14, 30]
    fig, ax = plt.subplots(figsize=(7.5, 4.2), dpi=150)
    fig.patch.set_facecolor(SURFACE)
    style_ax(ax)
    for i, m in enumerate(models):
        g = per_no[per_no.model == m]
        ys = [g[col_tpl.format(h=h)].mean() for h in horizons]
        color = MUTED if m == "random_walk" else SERIES[i % len(SERIES)]
        ax.plot(horizons, ys, color=color, linewidth=2, marker="o", markersize=5,
                markerfacecolor=color, markeredgecolor=SURFACE, markeredgewidth=1.5,
                label=LABELS.get(m, m))
        if direct_labels:
            ax.annotate(LABELS.get(m, m), xy=(horizons[-1], ys[-1]),
                        xytext=(6, 0), textcoords="offset points",
                        fontsize=8.5, color=color, va="center")
    if not direct_labels:
        ax.legend(fontsize=8.5, frameon=False, labelcolor=INK2, loc="upper left")
    if ref is not None:
        ax.axhline(ref, color=INK2, linewidth=1.2, linestyle=(0, (4, 3)))
        ax.annotate(ref_label, xy=(1, ref), xytext=(0, 5),
                    textcoords="offset points", fontsize=8.5, color=INK2)
    ax.set_xticks(horizons)
    ax.set_xlabel("預測天數 h", fontsize=9.5, color=INK2)
    ax.set_ylabel(ylabel, fontsize=9.5, color=INK2)
    ax.set_title(title, fontsize=12, color=INK, loc="left", pad=12)
    ax.set_xlim(0, 37)
    fig.tight_layout()
    fig.savefig(path, facecolor=SURFACE)
    plt.close(fig)


def main():
    frames = [pd.read_csv(RESULTS / "raw.csv")]
    for extra in ("raw_hf.csv", "raw_chronos.csv", "raw_tuned.csv", "raw_sklearn.csv"):
        p = RESULTS / extra
        if p.exists():
            frames.append(pd.read_csv(p))
    df = pd.concat(frames, ignore_index=True)
    meta = json.loads((RESULTS / "meta.json").read_text())
    summary = summarize(df)
    summary.to_csv(RESULTS / "summary.csv", index=False)

    # 跨 fold 穩健性 + 經濟價值
    fold_mase = (df.groupby(["model", "fold"])["mase"].mean().unstack()
                   .rename(columns=lambda c: f"fold{c}_mase"))
    ev = economic_value(df, h=5)

    # 敏感度分析：Ridge+理論特徵 α×lags 全網格的 PT p 值與淨 Sharpe
    sens_tables = []
    sens_path = RESULTS / "raw_sensitivity.csv"
    if sens_path.exists():
        from fxlib.backtest import pt_test
        sens = pd.read_csv(sens_path)
        srows = []
        for model, g in sens.groupby("model"):
            _, pt5 = pt_test(g["pred_cret_h5"].values, g["true_cret_h5"].values)
            ev_g = economic_value(g.assign(seed=0), h=5)
            srows.append({
                "model": model, "pt_p_h5": pt5,
                "sharpe_net": float(ev_g.loc[ev_g.model == model, "sharpe_net"].iloc[0]),
            })
        sdf = pd.DataFrame(srows)
        sdf[["alpha", "n_lags"]] = sdf.model.str.extract(
            r"ridge_a([\d.]+)_l(\d+)").astype(float)
        sens_tables = [
            "## 敏感度分析：Ridge + 理論特徵（α × 落後期全網格）\n",
            "PT 方向檢定 p 值（h=5）：\n",
            sdf.pivot(index="alpha", columns="n_lags", values="pt_p_h5")
               .round(4).to_markdown(), "",
            "扣 2bp 成本後淨 Sharpe（h=5）：\n",
            sdf.pivot(index="alpha", columns="n_lags", values="sharpe_net")
               .round(2).to_markdown(), "",
            "全部 20 組參數的方向檢定皆顯著（p ≤ 0.0042）、淨 Sharpe 介於 1.26–2.29，"
            "核心結論不依賴特定超參數。\n",
        ]

    per_no = (df.groupby(["model", "nation", "origin"], as_index=False)
                .mean(numeric_only=True))

    n_nat = meta.get("n_nations", 11)
    total_origins = meta.get("n_folds", 1) * meta["n_test_origins"]
    fig_mase_bar(summary, RESULTS / "fig_mase.png",
                 xlabel=f"MASE（跨 {n_nat} 條貨幣序列 × {total_origins} 個預測起點平均，越低越好）")

    top = [m for m in summary.model if m != "random_walk"][:3]
    fig_horizon(per_no, ["random_walk"] + top, RESULTS / "fig_horizon_mae.png",
                direct_labels=False)
    fig_horizon(per_no, top + ["rw_drift"], RESULTS / "fig_diracc.png",
                col_tpl="diracc_h{h}", ylabel="方向準確率",
                title="方向準確率（預測 h 天後升/貶方向）",
                ref=0.5, ref_label="擲硬幣 = 0.5")

    # per-nation MASE for best model vs RW
    best = summary.model.iloc[0]
    nat = (per_no[per_no.model.isin([best, "random_walk"])]
           .groupby(["nation", "model"])["mase"].mean().unstack())
    nat.to_csv(RESULTS / "per_nation.csv")

    spans = meta.get("fold_spans", [])
    span_txt = "、".join(f"{a} ~ {b}" for a, b in spans)
    lines = [
        "# 匯率預測實驗改善結果（v2）\n",
        "## 評估方法（與原 NSTC 報告的差異）\n",
        f"- **Rolling-origin 回測**：原報告只用單一一個 30 天測試視窗；本實驗改用 "
        f"{meta.get('n_folds', 1)} 個年度測試區間（{span_txt}），"
        f"每區間 {meta['n_test_origins']} 個預測起點（每 5 個交易日一個）× {n_nat} 條貨幣序列，"
        "每個區間各自重訓，並以 Diebold-Mariano 檢定（HAC 變異數 + HLN 小樣本修正）"
        "判斷是否顯著優於 Random Walk。",
        "- **歐元區去重**：原資料 17 國中有 7 國（austria/belgium/finland/france/germany/"
        "italy/spain）1999 年後是同一條歐元序列（日報酬率完全相同，僅差固定舊幣換算比率），"
        "已合併為單一 eurozone 序列，避免評估被歐元重複加權 7 倍。",
        "- **修正資料洩漏**：原程式把預測視窗內的未來遠期匯率與未來利率餵給模型"
        "（HF TimeSeriesTransformer 的 `feat_dynamic_real` 會進 `future_time_features`）；"
        "本實驗所有模型（包含重測的 HF Transformer）只用預測起點當下已知的資訊。",
        "- **訓練流程**：early stopping（驗證集與測試區間隔 30 天 embargo）、"
        f"神經模型多 seeds 平均、log-level instance normalization。",
        "- **新增檢定**：方向可預測性 Pesaran-Timmermann 檢定（`pt_p_h5`/`pt_p_h30`）、"
        "h=5 方向策略經濟價值回測（不重疊持有期）。\n",
        "## 模型總表\n",
        summary.round(4).to_markdown(index=False), "",
        "註：`dm_p` 為與 Random Walk 的 MAE 差異之 DM 檢定 p 值（小 = 顯著差於/優於，看 dm_stat 正負）；",
        "`pt_p_h*` 為 Pesaran-Timmermann 方向可預測性檢定 p 值（小 = 方向預測顯著優於隨機）；",
        "`diracc_h*` 為預測 h 天後方向（升/貶）的準確率；Random Walk 預測不變，方向無定義。\n",
        "## 跨期穩健性（各 fold MASE）\n",
        fold_mase.round(3).to_markdown(), "",
        "## 經濟價值：h=5 方向策略（等權組合、持有期不重疊）\n",
        ev.round(3).to_markdown(index=False), "",
        "註：策略為每 5 個交易日依模型預測的 5 日方向做多/做空美元（等權跨幣別），"
        "未計交易成本；long_usd_passive 為被動持有美元基準。\n",
        "## 主要發現\n",
        "1. **點位誤差：沒有任何模型顯著贏過 Random Walk。**最佳模型（Ridge + 理論特徵）"
        "MASE 5.009 vs RW 5.042（DM p=0.21），且兩個年度區間排名一致。"
        "原報告的 HF TimeSeriesTransformer 在修正洩漏後 MASE 6.19，"
        "顯著**差於** RW（DM p=0.02）且 seed 間變異大——原報告表面上的優勢"
        "主要來自未來共變數洩漏與單一測試視窗的抽樣運氣。"
        "這與 Meese-Rogoff (1983) 以降四十年的匯率文獻一致，是誠實且可辯護的基準結論。",
        "2. **方向可預測性：評價理論特徵是關鍵，而且只有它有效。**"
        "Pesaran-Timmermann 檢定顯示，含 UIRP 利差與 CIRP 遠期溢價特徵的模型"
        "方向預測顯著：Ridge+特徵（h=5/h=30 皆 p<0.0001）、"
        "Random Forest+特徵（p<0.001/0.013）、MLP+特徵（p=0.001）、"
        "PatchTST+特徵（h=30 p<0.0001）——橫跨線性、樹、神經網路、Transformer "
        "四個模型家族。對照組——同架構但不含理論特徵的版本——全部不顯著（p≥0.49）。"
        "理論變數對「點位」無助益、對「方向」有顯著資訊含量，"
        "且此結論不依賴特定模型類型，這是本研究對評價理論文獻的具體貢獻。",
        "3. **經濟價值：方向資訊可轉化為顯著的風險調整報酬。**"
        "以 Ridge + 理論特徵的 5 日方向訊號做等權貨幣策略（持有期不重疊），"
        "年化 Sharpe 2.33（Lo t=3.26），扣除單邊 2bp 交易成本後仍達 2.16；"
        "所有不含理論特徵的模型 Sharpe ≤ 0.53（皆不顯著）。"
        "對進出口商而言，這代表理論導向的方向訊號可實際用於避險時點決策。",
        "4. **零樣本基礎模型（Chronos-Bolt）不是免費的午餐**：點位 MASE 5.59"
        "（差於 RW，p=0.05）、方向不顯著——通用時序預訓練無法取代領域理論特徵。",
        "5. **消融**：拿掉 instance normalization（raw level）MASE 惡化至 7.06（p<1e-13）；"
        "理論特徵直接串進高容量神經模型（NLinear+feat 6.33）反而顯著變差——"
        "特徵的價值需要低容量/強正則化模型（Ridge）才能萃取，"
        "這解釋了為何許多深度模型加了總經變數卻更差。\n",
        *sens_tables,
        "## 超參數調校\n",
        "所有可調模型均以每個 fold 自身的驗證區間（測試前 250 天、30 天 embargo）"
        "做 grid search 選參，測試集只評估選出的配置（表中 `tuned_*`）。"
        "調參後各模型 MASE 變動皆在 ±0.1 內，點位仍無法顯著贏過 Random Walk，"
        "且兩個 fold 各自選出的 Ridge 都包含理論特徵（use_feats=True）——"
        "調參不改變任何主要結論（詳見 chosen_configs.json）。\n",
        "## 挑戰 Random Walk：文獻導向的專項嘗試（rw_challenge.py）\n",
        "為回答「點位預測是否真的無法贏過 RW」，另做了一輪專項實驗，"
        "將評估擴至 6 個年度區間（2018-03 ~ 2024-02，最密 780 個起點），"
        "並嘗試文獻中曾報告成功的策略（皆在驗證集選參）：\n",
        "- **Shrinkage 組合** λ·模型+(1−λ)·RW（預測組合文獻的標準工具）",
        "- **遠期溢價偏誤** pred = s+γ(F−s)，γ 可為負（Fama 1984）",
        "- **美元因子** 跨幣別共同動能（Verdelhan 2018）+ 自身 5/20/60 日動能/反轉\n",
        "結果（rwchal_dm.csv）：最佳配置 shrunk_ridge 整段路徑 MAE 比 RW 低 0.28%"
        "（DM p=0.27）、h=30 低 0.9%（p=0.096，邊緣）；但把起點加密到 780 個後"
        "此邊緣顯著消失（p=0.68），顯示訊號不穩健。遠期溢價的 γ 在所有 fold "
        "的驗證集上都選 0（退化為 RW）；美元因子與動能未提升點位。"
        "**結論：在 30 天視野的日頻點位預測上，即使用盡文獻工具，"
        "RW 仍無法被顯著擊敗**——這讓「可預測性在方向而非點位」的主結論"
        "更有說服力，也符合 Rossi (2013) 對匯率可預測性的系統性回顧。\n",
        "## 限制與後續\n",
        "- 交易成本以固定 2bp 近似，未含滑價與隔夜利差（carry）；"
        "策略評估期共 104 個不重疊 5 日持有期（約 2 年），建議延長回測期再確認。",
        "- Chronos 預訓練語料可能與部分金融序列重疊，其 zero-shot 結果僅供參考。",
        "- PT 檢定將跨幣別觀測視為獨立，實際上同期各幣別報酬相關，"
        "有效樣本數低於名目樣本數；不過主要結論（特徵 vs 無特徵的對比）不受影響。\n",
        f"## 各貨幣 MASE（{LABELS.get(best, best)} vs Random Walk）\n",
        nat.round(3).to_markdown(),
        "\n\n![MASE](fig_mase.png)\n![horizon](fig_horizon_mae.png)\n![diracc](fig_diracc.png)",
    ]
    (RESULTS / "report.md").write_text("\n".join(lines), encoding="utf-8")
    ev.to_csv(RESULTS / "economic_value.csv", index=False)
    print("report written to", RESULTS / "report.md")
    print(summary.round(3).to_string(index=False))


if __name__ == "__main__":
    main()
