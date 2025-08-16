import os, json, pathlib, pandas as pd, shutil
from azure.ai.ml import MLClient
from azure.identity import DefaultAzureCredential

BASE = pathlib.Path(__file__).resolve().parents[1]
ASSETS = BASE / "assets"
MODEL_DIR = BASE / "model"
CONFIG = json.load(open(BASE / "config.json"))

def ml():
    return MLClient(
        DefaultAzureCredential(),
        CONFIG["subscription_id"],
        CONFIG["resource_group"],
        CONFIG["workspace_name"],
    )

def main():
    client = ml()
    parent = client.jobs.get(CONFIG["automl_job_name"])
    best_id = parent.properties.get("best_child_run_id")
    if not best_id:
        raise RuntimeError("No best child found. Is the AutoML run complete?")
    best = client.jobs.get(best_id)
    metrics = best.properties.get("metrics") or {}
    (ASSETS / "metrics.json").write_text(json.dumps(metrics, indent=2))
    if metrics:
        pd.DataFrame([metrics]).to_csv(ASSETS / "metrics.csv", index=False)
    try:
        client.jobs.download(name=best.name, download_path=str(MODEL_DIR), output_name="model", all=False)
    except Exception as e:
        print("Download failed:", e)
    print("Done.")

if __name__ == "__main__":
    main()
