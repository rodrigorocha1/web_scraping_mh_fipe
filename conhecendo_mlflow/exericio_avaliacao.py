import os

import matplotlib.pyplot as plt
import mlflow.sklearn
import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split

MLFLOW_URI = "http://172.25.0.5:5000"
mlflow.set_tracking_uri(MLFLOW_URI)
mlflow.set_registry_uri(MLFLOW_URI)

EXPERIMENT_NAME = "experimento_hiperparametro"
mlflow.set_experiment(EXPERIMENT_NAME)

# ----------------------------
# 1. Criando um dataset de exemplo
# ----------------------------
np.random.seed(42)
n_samples = 100
X = np.random.rand(n_samples, 1) * 10  # Features
y = 2 * X.squeeze() + np.random.randn(n_samples) * 2  # Target com ruído

df = pd.DataFrame({"feature": X.squeeze(), "target": y})
train_df, eval_df = train_test_split(df, test_size=0.2, random_state=42)

# ----------------------------
# 2. Treinando modelo simples
# ----------------------------
model = LinearRegression()
model.fit(train_df[["feature"]], train_df["target"])


# ----------------------------
# 3. Função para visualização customizada
# ----------------------------
# ----------------------------
# 3. Função para visualização customizada
# ----------------------------
def create_custom_plot(eval_df, builtin_metrics, artifacts_dir=None):
    """
    Cria gráfico Predictions vs Targets e salva como artifact no MLflow.
    """
    import matplotlib.pyplot as plt
    import mlflow

    plt.figure(figsize=(8, 6))
    plt.scatter(eval_df["feature"], eval_df["target"], alpha=0.6, label="Target")
    plt.scatter(eval_df["feature"], model.predict(eval_df[["feature"]]),
                alpha=0.6, color='orange', label="Predictions")
    plt.xlabel("Feature")
    plt.ylabel("Target / Predictions")
    plt.title("Predictions vs Targets")
    plt.legend()

    # Salva o gráfico como artifact do run
    plot_path = "custom_plot.png"
    plt.savefig(plot_path)
    plt.close()

    # Loga o artifact no MLflow
    mlflow.log_artifact(plot_path)

    # Retorna dicionário esperado pelo evaluate
    return {"custom_plot": plot_path}



# ----------------------------
# 4. Avaliação com MLflow
# ----------------------------
mlflow.set_experiment("exemplo_custom_plot")

with mlflow.start_run() as run:
    # Log do modelo
    mlflow.sklearn.log_model(model, "linear_model")

    # URI do modelo para avaliação
    model_uri = f"runs:/{run.info.run_id}/linear_model"

    # Avaliação com artifact customizado
    result = mlflow.models.evaluate(
        model=model_uri,
        data=eval_df,
        targets="target",
        model_type="regressor",
        custom_artifacts=[create_custom_plot],
    )
