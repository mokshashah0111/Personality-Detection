import sys, pathlib, pandas as pd, mlflow

BASE = pathlib.Path(__file__).resolve().parents[1]

def load_model():
    for p in [BASE / "model", BASE / "model" / "mlflow-model"]:
        if (p / "MLmodel").exists():
            return mlflow.pyfunc.load_model(str(p))
    for p in BASE.rglob("MLmodel"):
        return mlflow.pyfunc.load_model(str(p.parent))
    raise FileNotFoundError("No MLflow model found.")

def main():
    if len(sys.argv) < 2:
        print("Usage: python scripts/predict.py <csv_path>")
        sys.exit(1)
    csv_path = pathlib.Path(sys.argv[1])
    df = pd.read_csv(csv_path)
    model = load_model()
    preds = model.predict(df)
    out = csv_path.parent / "predictions.csv"
    pd.DataFrame({"prediction": preds}).to_csv(out, index=False)
    print("Predictions saved to", out)

if __name__ == "__main__":
    main()
