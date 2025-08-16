# Azure AutoML – Personality Classification

This repository captures the **best model** from an Azure **Automated ML** run and shows its **metrics**, **artifacts**, and how to **run predictions** locally.

## Setup
```bash
python -m venv .venv
source .venv/bin/activate   # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```


## Export best model & metrics
```bash
python scripts/fetch_automl_best.py
```

## Run predictions
```bash
python scripts/predict.py assets/sample_input.csv
```
