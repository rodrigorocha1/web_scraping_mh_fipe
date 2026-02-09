import io
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from PIL import Image
import mlflow
from mlflow.metrics import MetricValue
from mlflow.models import make_metric, infer_signature
from sklearn.datasets import make_regression
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
from src_machine_learning.utils.mlflow_config import configurar_mlflow

# =========================
# Configura MLflow
# =========================
MLFLOW_URI = "http://172.25.0.5:5000"
EXPERIMENT_NAME = "regressao_underfit_overfit_buffer_final"
configurar_mlflow(experiment_name=EXPERIMENT_NAME, tracking_uri=MLFLOW_URI)

# =========================
# Dados
# =========================
X, y = make_regression(n_samples=1000, n_features=20, noise=15.0, random_state=42)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# =========================
# Métrica customizada
# =========================
def custom_metric_fn(predictions, targets, metrics):
    errors = predictions - targets
    custom_value = np.sum(np.where(errors > 0, errors * 2, errors))
    return MetricValue(
        aggregate_results={
            "custom_value": custom_value,
            "value_per_prediction": custom_value / len(predictions),
        }
    )

custom_metric = make_metric(
    eval_fn=custom_metric_fn,
    greater_is_better=False,
    name="custom_metric"
)

# =========================
# Funções de plot para buffer
# =========================
def plot_to_buffer(fig):
    buf = io.BytesIO()
    fig.savefig(buf, format="png")
    buf.seek(0)
    plt.close(fig)
    return buf

def plot_underfit_overfit(param_range, train_rmse, test_rmse):
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.plot(param_range, train_rmse, marker='o', label="Train RMSE")
    ax.plot(param_range, test_rmse, marker='o', label="Test RMSE")
    ax.set_xlabel("Max Depth")
    ax.set_ylabel("RMSE")
    ax.set_title("Underfitting vs Overfitting")
    ax.legend()
    ax.grid(True)
    return plot_to_buffer(fig)

def plot_predictions_vs_target(predictions, targets):
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.scatter(predictions, targets, alpha=0.5)
    ax.plot([targets.min(), targets.max()],
            [targets.min(), targets.max()],
            color='red', linestyle='--', label='Ideal')
    ax.set_xlabel("Predictions")
    ax.set_ylabel("Targets")
    ax.set_title("Predictions vs Targets")
    ax.legend()
    return plot_to_buffer(fig)

def log_plot_buffer(buf, artifact_name):
    """Converte buffer BytesIO para PIL.Image e loga no MLflow"""
    img = Image.open(buf)
    mlflow.log_image(img, artifact_name)

# =========================
# Avaliação de diferentes complexidades
# =========================
param_range = list(range(1, 21))
train_rmse_list = []
test_rmse_list = []

with mlflow.start_run():
    for depth in param_range:
        model = RandomForestRegressor(n_estimators=100, max_depth=depth, random_state=42)
        model.fit(X_train, y_train)

        y_train_pred = model.predict(X_train)
        y_test_pred = model.predict(X_test)

        train_rmse = np.sqrt(mean_squared_error(y_train, y_train_pred))
        test_rmse = np.sqrt(mean_squared_error(y_test, y_test_pred))

        train_rmse_list.append(train_rmse)
        test_rmse_list.append(test_rmse)

    # Melhor modelo (menor RMSE no teste)
    best_idx = np.argmin(test_rmse_list)
    best_model = RandomForestRegressor(n_estimators=100, max_depth=param_range[best_idx], random_state=42)
    best_model.fit(X_train, y_train)

    signature = infer_signature(X_test, best_model.predict(X_test))
    model_info = mlflow.sklearn.log_model(best_model, name="best_model", signature=signature)

    # Dataset de avaliação
    eval_data = pd.DataFrame(X_test)
    eval_data["target"] = y_test

    # =========================
    # Log plots usando buffer + PIL
    # =========================
    buf_under_over = plot_underfit_overfit(param_range, train_rmse_list, test_rmse_list)
    log_plot_buffer(buf_under_over, "underfit_overfit.png")

    predictions = best_model.predict(X_test)
    buf_pred_target = plot_predictions_vs_target(predictions, y_test)
    log_plot_buffer(buf_pred_target, "prediction_vs_target.png")

    # Avaliação MLflow
    result = mlflow.models.evaluate(
        model_info.model_uri,
        eval_data,
        targets="target",
        model_type="regressor",
        extra_metrics=[custom_metric],
    )

    print("Custom Value:", result.metrics['custom_metric/custom_value'])
