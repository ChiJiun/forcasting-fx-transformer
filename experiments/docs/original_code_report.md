# 原始程式報告

本報告詳細記錄改善計畫**開始前**兩個專案的程式結構、運作方式與發現的問題，作為修改前的基準文件。

---

## 第一部分：forcasting-fx-transformer（日頻深度學習專案）

此為 NSTC 報告對應的主程式碼，實作 Transformer 系列模型的匯率預測。

### 1. 專案結構

```
forcasting-fx-transformer/
├── nations/                  # 17 國日頻資料（每國一個 CSV）
│   ├── australia.csv … uk.csv
│   └── nations_concat.py     # 資料合併檢查小工具
├── transformer.py            # HF TimeSeriesTransformer（667 行）
├── informer.py               # HF Informer（669 行）
├── autoformer.py             # HF Autoformer（668 行）
├── PatchTST.py               # HF PatchTST（563 行）
├── smoke_test.py             # 快速冒煙測試（187 行）
├── run_all_models.ps1        # 批次執行腳本
├── informer_grid_experiment.py  # Informer 的 context/horizon 網格實驗
├── main.ipynb                # 探索用 notebook
├── requirements.txt
└── {transformer,informer,autoformer}_with_feature/  # 各模型輸出（圖+metric.csv）
```

### 2. 資料

每國 CSV 格式相同：`Date, FX, FFX, INTEREST, US_INTEREST`

| 欄位 | 意義 |
|---|---|
| FX | 即期匯率（每美元兌換的當地貨幣數） |
| FFX | 遠期匯率 |
| INTEREST | 本國短期利率 |
| US_INTEREST | 美國短期利率 |

- 期間：2004-07-26 ~ 2024-04-05，每國 5,140 個交易日，無缺值，日期完全對齊
- 17 國：Australia、Austria、Belgium、Canada、Denmark、Finland、France、Germany、Italy、Japan、Norway、South Korea、Spain、Sweden、Switzerland、Taiwan、UK

### 3. 主流程（以 transformer.py 為代表）

四個模型腳本是**同一份範本的複製修改**（差異只在 import 的 Config/Model 類別與少數參數），流程完全相同：

#### 3.1 資料集建構（`prepare_dataset()`，transformer.py:63-114）

每國切成三個版本，放進 HuggingFace `DatasetDict`：

```python
train:      target = fx[:-60]   # 去掉最後 60 天
validation: target = fx[:-30]   # 去掉最後 30 天
test:       target = fx         # 完整序列
```

同時把 `FFX, INTEREST, US_INTEREST` 三欄作為 `feat_dynamic_real` 附上（`feat_static_cat` 為國家編號）。註：程式中的 StandardScaler 特徵標準化被註解掉了（line 77-78），特徵以原始尺度進入模型。

#### 3.2 GluonTS 轉換鏈（`create_transformation()`，transformer.py:127-217）

`Chain` 依序做：缺值遮罩（AddObservedValuesIndicator）→ 日曆時間特徵（AddTimeFeatures，freq="1D"）→ age 特徵（AddAgeFeature，log scale）→ **把時間特徵與 `feat_dynamic_real` 垂直堆疊成 `time_features`**（VstackFeatures，line 197-205）→ 欄位改名為 HF 命名。

最後一步是關鍵：動態實數特徵被併入 `time_features`，而 HF 的預測介面會提供 `future_time_features`——詳見問題 4.1。

#### 3.3 視窗切割與 DataLoader（transformer.py:220-373）

- 訓練：`ExpectedNumInstanceSampler` 從序列中**隨機抽窗**，`InstanceSplitter` 切出 past（context 90 + max lag 37 = 127 天）與 future（30 天）
- 測試：`create_backtest_dataloader` 用 `ValidationSplitSampler` 取**最後一個視窗**
- 批次：`as_stacked_batches`，batch_size 128、每 epoch 200 batches

#### 3.4 模型設定（transformer.py:428-456）

```python
TimeSeriesTransformerConfig(
    prediction_length=30, context_length=90,
    lags_sequence=[1,2,3,4,5,6,7,11,12,13,23,24,25,35,36,37],
    num_time_features=len(time_features)+1,
    num_static_categorical_features=1,   # 國家 ID
    num_dynamic_real_features=3,         # FFX, INTEREST, US_INTEREST
    cardinality=[17], embedding_dimension=[1],
    encoder_layers=4, decoder_layers=4, d_model=32)
```

informer.py / autoformer.py 僅換成 `InformerConfig/InformerForPrediction`、`AutoformerConfig/AutoformerForPrediction`，超參數維持一致；PatchTST.py 使用 `PatchTSTConfig/PatchTSTForPrediction`（channel-independent patch 架構，無 lags/靜態特徵概念）。

#### 3.5 訓練迴圈（transformer.py:516-583)

- AdamW：lr 4e-4、betas (0.9, 0.95)、weight decay 0.1
- 固定 **120 epochs**，無 early stopping、無 checkpoint 選擇
- validation loss 用 `create_train_dataloader` 對 validation 序列**隨機抽窗**計算
- loss 累計有一個小 bug：`if idx % 100 == 0: train_loss += loss.item()`（每 100 批才累計一次，印出的 loss 只是抽樣）
- wandb 整合預設關閉

#### 3.6 評估與輸出（transformer.py:585-667)

- `model.generate()` 產生機率樣本（預設 100 條），取 median 作點預測
- 指標：MASE、sMAPE、MSE、MAE（HuggingFace `evaluate` 套件）
- 輸出：每國預測圖（median ± 0.5 std 區間）與 `metric.csv` 到 `<模型名>_with_feature/`

### 4. 發現的問題（依嚴重程度）

#### 4.1 未來共變數洩漏（嚴重）

`feat_dynamic_real`（FFX、兩國利率）經 VstackFeatures 併入 `time_features` 後，HF 模型在 `generate()` 時收到 `future_time_features`——**預測未來 30 天時，模型直接拿到那 30 天的遠期匯率與利率**。遠期匯率與即期匯率高度同步（相關性接近 1），等同提前看到答案。後續實驗證實：修掉洩漏後，該模型 MASE 從「看似優於 RW」變成顯著劣於 RW（6.19 vs 5.04，DM p=0.02）。

#### 4.2 單一測試視窗（嚴重）

測試集只有整段序列的**最後 30 天**（一次預測）。單一視窗的抽樣變異極大，模型排名幾乎由運氣決定——改用 104 個起點後，原報告的排名差異全部消失。

#### 4.3 驗證集與訓練集重疊（中）

validation 序列 = `fx[:-30]`，與 train = `fx[:-60]` 共享 98.8% 的資料；且 validation loss 是隨機抽窗（大多落在訓練期內）。validation loss 因此不是 hold-out 訊號，也沒有被用來選模型（無 early stopping）。

#### 4.4 歐元區重複計算（中）

Austria、Belgium、Finland、France、Germany、Italy、Spain 七國 1999 年後同屬歐元區，其 FX 只差固定的舊幣換算比率（實測日 log 報酬率最大差異 7.6e-5，為捨入雜訊）。17 國面板實際只有 11 條獨立貨幣，歐元被加權 7 倍。

#### 4.5 其他（輕）

- 無隨機種子控制、單次訓練，無法區分改善與雜訊
- 特徵標準化被註解掉，利率（~2）與匯率（台灣 ~33）尺度差 15 倍以上
- 無任何統計顯著性檢定
- 四份模型腳本高度重複（~660 行 × 4，僅數行不同），維護困難

---

## 第二部分：FX-predect-v2-main（月頻傳統 ML 專案）

早期專案，以**月頻**資料與評價理論學派分類做傳統機器學習實驗。**非 git repo**（應為 GitHub ZIP 下載副本）。

### 1. 專案結構

```
FX-predect-v2-main/
├── fx/                      # 核心套件：nation 類別 + 學派因子建構
│   └── data/organized_data/raw_data/   # 月頻原始資料
├── run.py                   # 主入口：18 國 × 視窗{20,30,60,120} × 落後{1,3,6,13} 平行執行
├── model_exp.py             # SVR / MLP 實驗（含 StandardScaler、離群值 clip）
├── random_forest_exp(.py)   # Random Forest 實驗
├── arima_exp(.py)           # ARIMA 基準
├── variable_selecting_exp(.py)  # 變數選擇實驗
├── SVR/                     # 每國 SVR 結果（18 國目錄 + models_result.csv）
├── result/                  # run.py 輸出（每組 win_size-lag 一個 CSV）
└── raw_data_visualization.py
```

### 2. 設計特點

- **學派架構**：README 列 CIRP、monetary、PPP、taylor（未完成）、UIRP 五個學派，每國以 `Nation` 類別封裝 `get_schools(win_or_recur, algo, win_size, lag)` 介面——按學派取因子、以 window 或 recursive 方式跑指定演算法
- 涵蓋 18 國（多一個獨立的 EuroZone 條目，處理方式比日頻專案合理）
- 前處理：`replace_inf_with_max_except_inf`、`clip_by_2_stander`（±2σ 截尾）、StandardScaler
- 以 `process_map` 多核平行掃 16 組（視窗×落後）設定

### 3. 與日頻專案的關係

此專案是評價理論特徵想法的來源（UIRP 利差、CIRP 遠期溢價等），但其月頻資料與學派因子**沒有**被日頻深度學習專案使用——日頻專案僅用了 FX/FFX/利率三個原始欄位。此斷層是改善計畫中「理論特徵」工作的出發點。

### 4. 觀察到的限制

- taylor 學派未完成（README 勾選清單）
- 評估方式與日頻專案不互通，結果無法直接比較
- 無版本控制（ZIP 副本），有遺失風險
