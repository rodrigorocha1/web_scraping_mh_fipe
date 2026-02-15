from mlflow.tracking import MlflowClient
import mlflow
from src_mlops.utils.mlflow_config import configurar_mlflow

MLFLOW_URI = "http://localhost:5000"

EXPERIMENT_NAME = f"modelo_pronto_votacao"
configurar_mlflow(experiment_name=EXPERIMENT_NAME, tracking_uri=MLFLOW_URI)

client = MlflowClient()

models = client.search_registered_models()

for model in models:
    print(f"Nome: {model.name}")
    print(f"Descrição: {model.description}")
    print("-" * 40)

run = mlflow.get_run("464858489e054717b030b8e66b869138")
print(run.info.artifact_uri)


run_id = "464858489e054717b030b8e66b869138"

client = mlflow.tracking.MlflowClient()
artifacts = client.list_artifacts(run_id)

for a in artifacts:
    print(a.path)