# FX Transformer Forecasting Project

這個專案提供多個時間序列預測模型，包含 Transformer、Informer、Autoformer 與 PatchTST。資料主要來自 `nations/` 目錄下的國家匯率 CSV 檔案，模型會直接讀取這些檔案進行訓練與評估。

## 環境需求

- Windows
- Python 3.12.3
- `pip`

## 安裝步驟

建議在專案根目錄執行以下指令：

```powershell
python -m venv .venv
.\.venv\Scripts\python.exe -m pip install --upgrade pip
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
```

如果你已經有現成的虛擬環境，也可以直接啟用它再安裝套件。

## 先做快速檢查

在正式訓練前，先跑 smoke test 來確認環境與資料是否正常：

```powershell
.\.venv\Scripts\python.exe smoke_test.py
```

這會對以下四個模型做一輪很小的訓練、驗證與生成測試：

- `transformer.py`
- `informer.py`
- `autoformer.py`
- `PatchTST.py`

## 執行完整訓練

每個模型都可以直接用 Python 執行：

```powershell
.\.venv\Scripts\python.exe transformer.py
.\.venv\Scripts\python.exe informer.py
.\.venv\Scripts\python.exe autoformer.py
.\.venv\Scripts\python.exe PatchTST.py
```

如果你想先看特定模型的效果，就只跑對應的腳本即可。

## 執行時間提醒

- 完整訓練在 CPU-only 環境會很久，可能需要數小時。
- `wandb` 預設在腳本中是關閉的，所以不需要先登入 Weights & Biases。

## 常見輸出

訓練過程與評估結果會依腳本寫入各自的輸出位置，相關圖表或結果檔通常會放在對應模型的資料夾中，例如 `transformer/`、`informer/`、`autoformer/`、`transformer_with_feature/` 等。

## 建議流程

1. 建立並啟用虛擬環境。
2. 安裝 `requirements.txt`。
3. 先跑 `smoke_test.py`。
4. 再執行你要的模型腳本。

如果你只是想確認專案是否能正常跑，先做 smoke test 就夠了。