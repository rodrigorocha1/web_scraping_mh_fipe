from mlflow.tracking import MlflowClient
import mlflow

mlflow.set_tracking_uri("http://localhost:5000")

client = MlflowClient()

for model in client.search_registered_models():
    print("Model:", model.name)

    for version in model.latest_versions:
        print("  Version:", version.version)
        print("  Stage:", version.current_stage)
        print("  Run ID:", version.run_id)
