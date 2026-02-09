import mlflow
from mlflow.models import infer_signature
from sklearn.datasets import load_breast_cancer
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.model_selection import train_test_split

# --------------------------
# Configuração do MLflow
# --------------------------
MLFLOW_URI = "http://172.25.0.5:5000"
mlflow.set_tracking_uri(MLFLOW_URI)
mlflow.set_registry_uri(MLFLOW_URI)

EXPERIMENT_NAME = "wine_random_forest_v2"
mlflow.set_experiment(EXPERIMENT_NAME)

# Ativa autologging do scikit-learn, mas sem log automático de modelos
# porque vamos logar manualmente
mlflow.sklearn.autolog()

# --------------------------
# Preparação dos dados
# --------------------------
cancer = load_breast_cancer()
X_train, X_test, y_train, y_test = train_test_split(
    cancer.data, cancer.target, test_size=0.2, random_state=42
)

# --------------------------
# Treinamento e registro
# --------------------------
with mlflow.start_run():
    # Cria e treina o modelo
    model = GradientBoostingClassifier(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)

    # Previsões e assinatura do modelo
    y_pred = model.predict(X_test)
    signature = infer_signature(X_test, y_pred)

    # Log manual do modelo no MLflow
    mlflow.sklearn.log_model(
        sk_model=model,
        artifact_path="model",                # pasta dentro de /mlflow/artifacts
        signature=signature,
        input_example=X_test[:5],
        registered_model_name="wine_random_forest"
    )

    print(f"✅ Modelo registrado no MLflow! Run: {mlflow.active_run().info.run_id}")
