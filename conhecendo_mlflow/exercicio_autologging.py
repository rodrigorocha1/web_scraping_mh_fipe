from io import BytesIO

import matplotlib.pyplot as plt
import mlflow.sklearn
from PIL import Image  # ✅ Necessário para MLflow
from sklearn.datasets import make_regression
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.model_selection import train_test_split

from src_machine_learning.utils.mlflow_config import configurar_mlflow

# =========================
# Configurar MLflow
# =========================
configurar_mlflow(
    experiment_name='regressao_completa_w',
    tracking_uri='http://172.25.0.5:5000'
)

# =========================
# Gerar dados de regressão
# =========================
X, y = make_regression(
    n_samples=500,
    n_features=5,
    noise=20,
    random_state=42
)

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# =========================
# Iniciar Experimento MLflow
# =========================
with mlflow.start_run(run_name="random_forest_regressor") as run:
    # Treinar modelo
    model = RandomForestRegressor(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)

    # Fazer previsões
    y_pred_train = model.predict(X_train)
    y_pred_test = model.predict(X_test)

    # Calcular métricas
    rmse_train = mean_squared_error(y_train, y_pred_train)
    rmse_test = mean_squared_error(y_test, y_pred_test)
    r2_train = r2_score(y_train, y_pred_train)
    r2_test = r2_score(y_test, y_pred_test)

    print(f"RMSE treino: {rmse_train:.2f}, RMSE teste: {rmse_test:.2f}")
    print(f"R2 treino: {r2_train:.2f}, R2 teste: {r2_test:.2f}")

    # Logar métricas no MLflow
    mlflow.log_metric("rmse_train", rmse_train)
    mlflow.log_metric("rmse_test", rmse_test)
    mlflow.log_metric("r2_train", r2_train)
    mlflow.log_metric("r2_test", r2_test)

    # Logar o modelo
    mlflow.sklearn.log_model(model, "random_forest_model")

    # =========================
    # Gráfico: Real vs Predito
    # =========================
    fig, ax = plt.subplots(figsize=(8, 6))
    ax.scatter(y_test, y_pred_test, alpha=0.7)
    ax.plot([y.min(), y.max()], [y.min(), y.max()], 'r--', linewidth=2)
    ax.set_xlabel("Valores reais")
    ax.set_ylabel("Previsões")
    ax.set_title("Random Forest Regressor: Real vs Predito")
    ax.grid(True)

    # Converter para PIL Image e logar
    buf = BytesIO()
    fig.savefig(buf, format="png")
    buf.seek(0)
    img = Image.open(buf)
    mlflow.log_image(img, "real_vs_predito.png")
    plt.close(fig)

    # =========================
    # Gráfico: Resíduos
    # =========================
    residuals = y_test - y_pred_test
    fig, ax = plt.subplots(figsize=(8, 6))
    ax.scatter(y_pred_test, residuals, alpha=0.7)
    ax.hlines(0, xmin=y_pred_test.min(), xmax=y_pred_test.max(), colors='r', linestyles='dashed')
    ax.set_xlabel("Previsões")
    ax.set_ylabel("Resíduos")
    ax.set_title("Resíduos vs Previsões")
    ax.grid(True)

    buf = BytesIO()
    fig.savefig(buf, format="png")
    buf.seek(0)
    img = Image.open(buf)
    mlflow.log_image(img, "residuos.png")
    plt.close(fig)
