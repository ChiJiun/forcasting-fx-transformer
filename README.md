# FX Forecasting with Economic Theory and Modern Time-Series Models

本專案研究**多國匯率預測（FX forecasting）**，比較 Random Walk、傳統 machine learning、線性 time-series model、Transformer 系列與 foundation model，並檢驗 **UIRP / CIRP 等經濟理論變數是否能提升匯率預測能力**。

目前 repo 同時保留：

1. **早期模型實驗**：Transformer、Informer、Autoformer、PatchTST 等原始訓練程式與輸出。
2. **最終嚴謹評估框架**：位於 `experiments/`，修正早期實驗中的資料洩漏與評估設計問題，加入 rolling-origin backtest、統計檢定、超參數敏感度分析與經濟價值回測。

> 若要了解本專案的最終研究結論，請以 `experiments/` 的結果為主，而不是早期單一測試視窗的模型輸出。

---

## Research Context

本專案延伸自 NSTC 大專學生研究計畫：

**「基於評價理論與圖神經網路預測匯率之研究」**

NSTC 113-2813-C-008-073-E

後續實驗聚焦於：

- 經濟理論變數是否提供可用的匯率資訊；
- Transformer 類模型是否能穩健優於 Random Walk；
- 匯率的 **point forecast** 與 **direction forecast** 是否呈現不同的可預測性；
- 預測訊號是否具備統計與經濟意義。

---

## Final Research Design

最終評估框架位於：

```text
experiments/
```

核心設定如下：

| 項目 | 設定 |
|---|---|
| 貨幣序列 | 11 條獨立貨幣序列 |
| Eurozone | 將重複的歐元區序列去重 |
| Context length | 90 trading days |
| Forecast horizon | 30 trading days |
| Backtest | 2 個年度區間 × 每區間 52 個 rolling origins |
| Test stride | 5 trading days |
| Neural seeds | 3 |
| Validation | 測試區間前 250 天 |
| Leakage control | train/validation target 與 test 之間保留 30 天 embargo |
| Point metrics | MASE、sMAPE、MAE、MSE |
| Direction metrics | h=1/5/14/30 directional accuracy |
| Statistical tests | Diebold-Mariano、Pesaran-Timmermann |
| Economic evaluation | 5-day direction strategy、Sharpe、transaction cost adjustment |

所有最終模型只使用**預測起點當下可取得的資訊**，避免把未來 FFX / interest rate 誤送入 future covariates。

---

## Economic-Theory Features

目前主要理論導向特徵包括：

- `log_ret`：FX log return
- `rate_diff`：本國利率 − 美國利率，對應 UIRP 核心資訊
- `fwd_premium`：`(FFX - FX) / FX`，對應 CIRP / forward premium
- `rate_diff_chg5`：利差 5 日變化

實作位置：

```text
experiments/fxlib/data.py
```

---

## Models

最終框架比較的模型包含：

### Baselines
- Random Walk
- Random Walk + drift

### Statistical / Machine Learning
- Ridge regression
- Ridge + economic-theory features
- Random Forest
- Random Forest + economic-theory features
- MLP
- MLP + economic-theory features

### Modern Time-Series Models
- NLinear
- DLinear
- PatchTST
- PatchTST + economic-theory features
- tuned NLinear / DLinear / PatchTST

### Transformer / Foundation Models
- HF TimeSeriesTransformer
- Chronos-Bolt small (zero-shot)

Repo 根目錄另外保留較早期的：

- `transformer.py`
- `informer.py`
- `autoformer.py`
- `PatchTST.py`

這些檔案主要用於保留研究歷程與早期模型實作。

---

## Final Results

### 1. Point Forecast: no model robustly beats Random Walk

跨 **11 貨幣 × 104 rolling origins**：

| Model | MASE | DM p-value vs Random Walk |
|---|---:|---:|
| Ridge + theory features | **5.009** | 0.210 |
| Tuned Ridge | 5.016 | 0.833 |
| Ridge (returns only) | 5.026 | 0.200 |
| Random Walk | 5.042 | — |
| Tuned PatchTST | 5.055 | 0.890 |
| HF TimeSeriesTransformer | 6.190 | 0.024 |

雖然 Ridge + theory features 的平均 MASE 略低於 Random Walk，但差異**未達統計顯著**。

修正 future-covariate leakage 後，原本的 HF TimeSeriesTransformer 反而顯著差於 Random Walk。

因此目前結果不支持「Transformer 可以穩健擊敗 Random Walk」這個結論。

---

### 2. Direction Forecast: theory features contain useful information

`Ridge + theory features`：

- 5-day directional accuracy：**56.12%**
- Pesaran-Timmermann p-value (h=5)：約 **0.00002**
- h=30 同樣具有顯著方向可預測性

重要的是，同架構移除 theory features 後，方向預測顯著性大幅下降。

目前最主要的研究發現是：

> **經濟理論特徵對匯率 point forecast 的改善有限，但對方向預測具有明顯資訊含量。**

---

### 3. Economic Value

以 Ridge + theory features 的 5-day direction signal 建立等權貨幣策略：

| Metric | Result |
|---|---:|
| Annualized return | **8.9%** |
| Annualized volatility | 3.8% |
| Sharpe ratio | **2.33** |
| Lo Sharpe t-statistic | **3.26** |
| Net Sharpe after 2 bp one-way cost | **2.16** |

此結果表示方向訊號具有潛在經濟價值，但仍應搭配下方的研究限制解讀。

---

## Why the Final Framework Is Different from the Early Experiments

研究後期重新檢查早期程式與結果後，發現數個會影響結論的重要問題：

1. **Future-covariate leakage**
   早期 HF TimeSeriesTransformer 將未來區間的 FFX / interest rate 作為 dynamic features，會高估模型效果。

2. **Single test window**
   只用最後 30 天做一次測試，結果容易受單一市場狀態影響。

3. **Validation overlap**
   早期 validation 並非完全獨立的 hold-out signal。

4. **Eurozone duplication**
   多個歐元區國家在 1999 年後實際上對應同一條 EUR 序列，若分開計算會重複加權。

5. **No formal significance tests**
   早期只比較平均誤差，無法判斷差異是否只是抽樣波動。

因此最終研究改採 leakage-free rolling-origin backtest，並增加統計檢定與 robustness analysis。

---

## Robustness Checks

目前已完成：

- 2 個不同年度測試區間；
- validation-only hyperparameter selection；
- neural models multi-seed evaluation；
- Ridge 超參數 sensitivity grid；
- Diebold-Mariano test vs Random Walk；
- Pesaran-Timmermann directional predictability test；
- 5-day economic-value backtest；
- 交易成本調整；
- 額外 Random Walk challenge：
  - shrinkage forecast combination
  - forward-premium bias
  - dollar factor
  - momentum / reversal
  - 更密集 rolling origins

即使擴充 Random Walk challenge，point forecast 仍沒有穩健、顯著擊敗 Random Walk。

---

## Repository Structure

```text
.
├── transformer.py
├── informer.py
├── autoformer.py
├── PatchTST.py
├── nations/                         # 各國整理後資料
├── data/                            # 原始 / 中間資料
├── transformer/                     # 早期 Transformer outputs
├── informer/                        # 早期 Informer outputs
├── autoformer/                      # 早期 Autoformer outputs
├── *_with_feature/                  # 早期加入額外特徵的實驗
│
└── experiments/
    ├── fxlib/
    │   ├── data.py                  # leakage-free data / features
    │   ├── models.py                # 統一模型介面
    │   └── backtest.py              # metrics / DM / PT / economic value
    ├── run_experiments.py           # 主 rolling-origin 實驗
    ├── sklearn_exp.py               # Random Forest / MLP
    ├── hf_transformer_exp.py        # HF Transformer 無洩漏重測
    ├── chronos_exp.py               # Chronos-Bolt zero-shot
    ├── tune_experiments.py          # validation-based tuning
    ├── rw_challenge.py              # Random Walk 專項挑戰
    ├── make_report.py               # 彙整表格與圖表
    ├── docs/
    │   ├── original_code_report.md
    │   ├── modified_code_report.md
    │   └── performance_report.md
    └── results/
        ├── full_project_report.md
        ├── report.md
        ├── summary.csv
        ├── economic_value.csv
        └── ...
```

---

## Key Reports

建議依下列順序閱讀：

1. **完整研究報告**
   [`experiments/results/full_project_report.md`](experiments/results/full_project_report.md)

2. **詳細模型表現與所有主要數值**
   [`experiments/docs/performance_report.md`](experiments/docs/performance_report.md)

3. **自動產生的結果報告**
   [`experiments/results/report.md`](experiments/results/report.md)

4. **原始程式問題分析**
   [`experiments/docs/original_code_report.md`](experiments/docs/original_code_report.md)

5. **改善後程式架構說明**
   [`experiments/docs/modified_code_report.md`](experiments/docs/modified_code_report.md)

---

## Environment

主要環境：

- Windows
- Python 3.12.3
- pip

建立環境：

```powershell
python -m venv .venv
.\.venv\Scripts\python.exe -m pip install --upgrade pip
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
```

---

## Quick Smoke Test

先確認原始四個 time-series model 可以正常載入與執行：

```powershell
.\.venv\Scripts\python.exe smoke_test.py
```

Smoke test 涵蓋：

- Transformer
- Informer
- Autoformer
- PatchTST

---

## Reproduce the Final Experiments

以下為主要實驗流程：

```powershell
.\.venv\Scripts\python.exe experiments\run_experiments.py
.\.venv\Scripts\python.exe experiments\sklearn_exp.py
.\.venv\Scripts\python.exe experiments\hf_transformer_exp.py
.\.venv\Scripts\python.exe experiments\chronos_exp.py
.\.venv\Scripts\python.exe experiments\tune_experiments.py
.\.venv\Scripts\python.exe experiments\rw_challenge.py
.\.venv\Scripts\python.exe experiments\make_report.py
```

大部分實驗腳本亦提供 `--smoke` 模式，可先用較小規模確認環境。

完整 neural-model training 在 CPU-only 環境可能需要數小時。

---

## Research Limitations

目前結果仍有以下限制：

- 經濟價值回測以固定 **2 bp one-way transaction cost** 近似，未完整納入滑價與 carry。
- 主要 economic-value period 為約兩年的 104 個不重疊 5-day holding periods。
- Pesaran-Timmermann test 將跨貨幣觀測近似視為獨立，但實際上同日貨幣報酬存在 cross-sectional correlation，因此極小 p-value 應保守解讀。
- Chronos 的預訓練資料可能包含部分金融時間序列，因此 zero-shot 結果僅作 baseline。
- 本研究結果支持「directional information」而非「穩健的 point-forecast superiority」。

若進一步投稿，可優先考慮：

- 延長至更多市場週期；
- 使用 cluster-aware / block-bootstrap inference；
- 將 carry 與更完整交易成本納入 economic evaluation；
- 依真正資料發布時點加入月頻 PPP / Taylor-rule 等 macro features。

---

## Related Repository

早期的資料處理、經濟理論變數與 classical ML 實驗另見：

**[ChiJiun/FX-predect-v2](https://github.com/ChiJiun/FX-predect-v2)**

該 repo 可視為本研究的前期 exploratory stage；本 repo 的 `experiments/` 為目前最終、較嚴謹的 evaluation framework。

---

## Status

目前核心研究流程、模型比較、rolling-origin evaluation、統計檢定、robustness analysis 與結果報告皆已完成。

**專案現階段可作為研究結案成果；若要進一步投稿，建議針對 inference 與更長期 backtest 再強化。**
