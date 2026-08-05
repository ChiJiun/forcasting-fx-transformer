# 修改後程式報告（experiments/ 新框架）

本報告詳細記錄改善計畫新增的程式：架構設計、每個模組的職責與關鍵實作、資料流、以及完整重現方式。所有新程式位於 `forcasting-fx-transformer/experiments/`，**未修改**原始四個模型腳本（原程式保留為對照）。

---

## 1. 設計原則

1. **無洩漏**：任何模型在預測起點 t 之後的資訊（未來共變數、未來價格）都不可見；訓練/驗證視窗的預測目標與測試區之間保留 30 天 embargo。
2. **多起點評估**：以 rolling-origin 回測取代單一測試視窗，所有比較附統計檢定。
3. **統一介面**：所有模型實作 `fit(train_windows, val_windows)` / `predict(windows) -> [M, H]`，共用同一組視窗與指標程式。
4. **選參紀律**：超參數只在驗證區間選擇，測試集只評估選出的配置。
5. **可重現**：固定 seeds、結果全部落地為 CSV、每個腳本支援 `--smoke` 冒煙模式。

## 2. 目錄結構

```
experiments/
├── fxlib/                      # 共用函式庫
│   ├── data.py                 # 資料載入、歐元去重、理論特徵、視窗切割
│   ├── models.py               # 模型動物園 + 共用訓練器
│   └── backtest.py             # 指標、DM/PT 檢定、經濟價值回測
├── run_experiments.py          # 主實驗矩陣
├── hf_transformer_exp.py       # 原報告模型無洩漏重測
├── chronos_exp.py              # Chronos-Bolt 零樣本
├── sklearn_exp.py              # RF/MLP（原報告方法）補測
├── tune_experiments.py         # 超參數調校 + 敏感度
├── rw_challenge.py             # 「打敗 RW」專項實驗
├── make_report.py              # 彙整結果、產圖、寫報告
├── docs/                       # 本文件與原始程式報告
└── results/                    # 所有輸出（raw*.csv、summary、圖、報告）
```

## 3. fxlib 模組詳解

### 3.1 data.py

- `EURO_MEMBERS`：7 個歐元區國家集合。`load_panel(dedup_euro=True)` 只保留 germany 並改名 `eurozone`（合併前已驗證 7 國日報酬率最大差異 7.6e-5，屬捨入雜訊）。回傳 11 條貨幣 × 5140 天的 `fx [N,T]` 與特徵 `feats [N,T,4]`。
- 特徵（皆為當日已知，命名於 `FEATURE_NAMES`）：
  1. `log_ret`：日 log 報酬
  2. `rate_diff`：本國利率 − 美國利率（**UIRP** 核心變數）
  3. `fwd_premium`：(FFX − FX)/FX（**CIRP** 遠期溢價）
  4. `rate_diff_chg5`：利差 5 日變化
- `make_windows(fx, feats, origins, L, H)`：以「預測起點 = 最後已知日」切窗，回傳 `ctx_level [M,L]`、`ctx_feat [M,L,F]`、`tgt_level [M,H]`、`nation_id [M]`，M = 起點數 × 11。視窗排序固定為 [origin × nation]，下游（如美元因子）依賴此順序。

### 3.2 models.py

**基準**：`NaiveRW`（pred = 最後價位）、`Drift`（context 平均日變化外推）。

**RidgeReturns**（`use_feats` 切換理論特徵）：設計矩陣 = 最近 20 日 log 報酬（+ 標準化的 rate_diff / fwd_premium / rate_diff_chg5，取起點當日值）；目標 = h=1..30 的**累積 log 報酬**（多輸出直接預測，非遞迴）；還原時 `exp(log(s_t) + ŷ)`。

**神經網路**（`_NLinearNet` / `_DLinearNet` / `_PatchTSTNet`）：
- NLinear：log-level 減最後值 → 單層 Linear L→H（+feat 版把 [L,F] 特徵攤平串接）
- DLinear：replicate-pad 移動平均分解 trend/seasonal，各一個 Linear 後相加
- PatchTST-lite：patch 長 10 → Linear embedding + 可學位置編碼 → 2 層 TransformerEncoder（d=64, pre-norm）→ flatten head

**TorchForecaster**（共用訓練器）：
- log-level instance normalization（減 context 最後值；`normalize=False` 為消融）
- L1 loss、AdamW（lr 1e-3、wd 1e-4）、batch 512、最多 40 epochs
- early stopping：驗證 loss patience 6，**還原最佳 checkpoint**
- `torch.manual_seed(seed)` 完整控制隨機性；特徵以訓練期均值/標準差標準化

`build_models(seed)` 回傳 10 個變體（含 raw-level 與 +feat 消融）。

### 3.3 backtest.py

- `metric_rows()`：每 (nation, origin) 一列——MAE/MSE/sMAPE/MASE、方向準確率與帶號累積報酬 `pred_cret_h*`/`true_cret_h*`（h ∈ {1,5,14,30}）。
- `mase_scales()`：MASE 分母 = 訓練期 naive 一步 MAE（每貨幣）。
- `dm_test(a, b, max_lag)`：Diebold-Mariano，Bartlett-kernel HAC 變異數（處理重疊視窗自相關）+ Harvey-Leybourne-Newbold 小樣本修正；樣本過少時 lag 自動縮減（`min(max_lag, T//4)`）。
- `pt_test()`：Pesaran-Timmermann (1992) 方向可預測性檢定（考慮方向邊際機率，比裸方向準確率嚴格——例如永遠猜同方向的模型在 PT 下不顯著）。
- `economic_value(df, h=5, cost_oneway=2bp)`：每 5 日依預測方向持有（不重疊）、等權跨幣別組合；輸出年化報酬/波動/Sharpe、Lo (2002) 標準誤 t 值、扣部位變動成本後淨 Sharpe。
- `summarize()`：多 seed 平均 → 模型層級總表（含 DM 與 PT p 值）。

## 4. 實驗腳本詳解

### 4.1 run_experiments.py（主矩陣）

切分方案（每 fold 獨立）：

```
fold f 的測試起點：52 個、stride 5（跨約一年），最後起點 = T-1-30-f×260
train_cutoff   = 測試首起點 − 30           （embargo）
驗證起點       = train_cutoff−30−250 … train_cutoff−30，stride 5（50 個）
訓練起點       = 90−1 … 驗證首起點−30，stride 2
```

2 folds（2022、2023 年度）× 10 模型 × 3 seeds（確定性模型跑 1 次）→ `results/raw.csv` + `meta.json`。

### 4.2 hf_transformer_exp.py（原模型無洩漏重測）

與原 transformer.py 的三個關鍵差異：
1. `num_dynamic_real_features=0`——不給任何未來不可知的共變數；時間特徵只有日曆（dow/dom/doy 依 gluonts 慣例縮放）與 log-age，這些在未來視窗是真正已知的
2. **繞過 gluonts**：直接以 `make_windows`（L=127=context 90+max lag 37）建 `past_values / past_time_features / future_time_features / static_categorical_features` 張量餵 HF 模型，與其他模型共用同一組起點與 embargo
3. early stopping（patience 4）+ 最佳 checkpoint；其餘超參數沿用原報告（d32、4+4 層、lags、AdamW 4e-4/wd 0.1）以確保是「同一個模型」的公平重測

輸出 `raw_hf.csv`（2 folds × 2 seeds）。

### 4.3 chronos_exp.py（零樣本基礎模型）

`amazon/chronos-bolt-small`，context 512 天，`predict_quantiles` 取 median，完全不訓練。標註 caveat：預訓練語料可能與金融序列重疊。輸出 `raw_chronos.csv`。

### 4.4 sklearn_exp.py（原報告 RF/MLP 補測）

RandomForestRegressor（n_estimators=100，同原報告預設）與 MLPRegressor（64×64、Adam、early stopping），輸入與 Ridge 相同的設計矩陣（±理論特徵）、多輸出累積報酬目標。**刻意不 import torch**（獨立實作設計矩陣），因 venv 的 torch DLL 曾出現載入失敗。輸出 `raw_sklearn.csv`（2 folds × 3 seeds × 4 變體）。

### 4.5 tune_experiments.py（調參）

每 fold 流程：對每組候選（Ridge 40 組：α×{5,10,20,40}lags×±feat；NLinear 8 組：context×lr；DLinear 9 組：context×kernel；PatchTST 11 組：d/layers/patch/lr/context）→ 訓練 → **驗證起點 MAE** → 選最低者 → 只把選出配置放到測試集（`tuned_*`，神經模型 3 seeds）。另把 Ridge+feat 全網格的測試表現存成 `raw_sensitivity.csv`（敏感度分析，非選參依據）。選中配置記錄於 `chosen_configs.json`。

### 4.6 rw_challenge.py（打敗 RW 專項）

6 個年度 fold（2018-03 ~ 2024-02），`--stride` 可調（5 → 312 起點；2 → 780 起點）。候選（皆驗證集選參）：
- `shrunk_ridge`：λ·Ridge+feat + (1−λ)·RW，λ ∈ {0.1..1.0}
- `fwd_bias`：s + γ(h/30)(F−s)，γ ∈ {−2..2}（γ<0 = Fama 遠期溢價偏誤方向）
- `RidgeRich`：再加自身 5/20/60 日動能與**美元因子**（同起點跨幣別平均動能，利用視窗 [origin×nation] 排序 reshape 取均值）
- `combo`：shrunk 版本等權平均

DM 檢定分整段路徑 / h=1 / h=5 / h=30，HAC lag 隨 stride 調整（`max(6, H//stride+1)`）。輸出 `raw_rwchal.csv`、`rwchal_dm.csv`。

### 4.7 make_report.py（彙整）

合併 `raw.csv + raw_hf + raw_chronos + raw_tuned + raw_sklearn` → `summarize()` 總表、跨 fold MASE、經濟價值表、敏感度網格 → 產出三張圖（MASE 橫條、誤差-視野曲線、方向準確率曲線；Microsoft JhengHei 中文字型、依 dataviz 規範的色票與樣式）→ 寫 `report.md`。

## 5. 資料流總覽

```
nations/*.csv
   └─ load_panel（歐元去重、理論特徵）
        └─ make_windows（per-fold 訓練/驗證/測試起點，embargo 30 天）
             ├─ run_experiments   ─┐
             ├─ hf_transformer_exp ├─ raw*.csv ─ make_report ─ report.md + 圖表
             ├─ chronos_exp        │
             ├─ sklearn_exp        │
             ├─ tune_experiments  ─┘（+ chosen_configs.json / raw_sensitivity.csv）
             └─ rw_challenge ────── rwchal_dm.csv
```

## 6. 重現方式

```
.\.venv\Scripts\python.exe experiments\run_experiments.py     # ~40 分（CPU）
.\.venv\Scripts\python.exe experiments\hf_transformer_exp.py  # ~2-3 小時
.\.venv\Scripts\python.exe experiments\chronos_exp.py         # ~1 分
.\.venv\Scripts\python.exe experiments\sklearn_exp.py         # ~15 分
.\.venv\Scripts\python.exe experiments\tune_experiments.py    # ~1 小時
.\.venv\Scripts\python.exe experiments\rw_challenge.py        # ~1 分
.\.venv\Scripts\python.exe experiments\make_report.py         # 秒級
```

每個腳本可先加 `--smoke` 驗證環境。新增依賴：`tabulate`（表格輸出）、`chronos-forecasting`（零樣本 baseline）。

## 7. 與原始程式的對照摘要

| 面向 | 原始程式 | 修改後 |
|---|---|---|
| 測試視窗 | 1 個（最後 30 天） | 104 個起點（主矩陣）/ 780 個（挑戰賽） |
| 未來共變數 | 洩漏（feat_dynamic_real 進未來視窗） | 全部只用起點當下已知資訊 |
| 驗證集 | 與訓練重疊、未用於選模 | 獨立區間 + embargo、early stopping + 選 checkpoint |
| 歐元區 | 7 國重複計算 | 合併為單一序列 |
| 檢定 | 無 | DM（HAC+HLN）、PT、Lo Sharpe SE |
| 隨機性 | 無 seed、單次 | 固定 seeds × 3、平均 |
| 選參 | 無 | 驗證集 grid search + 測試集敏感度分析 |
| 程式重複 | 4×660 行近複製 | 共用 fxlib，模型以類別註冊 |
