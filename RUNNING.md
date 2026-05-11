# Running the Project

This project was verified on Windows with Python 3.12.3.

## Setup

```powershell
python -m venv .venv
.\.venv\Scripts\python.exe -m pip install --upgrade pip
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
```

## Quick Environment Check

Run the smoke test before launching a full training job:

```powershell
.\.venv\Scripts\python.exe smoke_test.py
```

This runs one small train/validation/generation pass for:

- `transformer.py`
- `informer.py`
- `autoformer.py`
- `PatchTST.py`

## Full Training

Each full script can be run directly:

```powershell
.\.venv\Scripts\python.exe transformer.py
.\.venv\Scripts\python.exe informer.py
.\.venv\Scripts\python.exe autoformer.py
.\.venv\Scripts\python.exe PatchTST.py
```

The default full training settings are CPU-heavy. On a CPU-only environment, a full run may take many hours.

Weights & Biases logging is disabled by default in the scripts so the run does not block on login.
