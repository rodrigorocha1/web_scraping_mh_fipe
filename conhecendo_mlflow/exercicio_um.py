import mlflow
from sklearn.ensemble import RandomForestClassifier
from sklearn.datasets import load_wine
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from src_machine_learning.utils.mlflow_config import configurar_mlflow

# =========================
# Configurar MLflow
# =========================
configurar_mlflow(
    experiment_name='regressao_completa_pipeline',
    tracking_uri='http://172.25.0.5:5000'
)

# Habilitar autolog do sklearn
mlflow.sklearn.autolog()

# =========================
# Carregar dados
# =========================
wine = load_wine()
X_train, X_test, y_train, y_test = train_test_split(
    wine.data, wine.target, test_size=0.2, random_state=42
)

# =========================
# Criar Pipeline
# =========================
pipeline = Pipeline([
    ('scaler', StandardScaler()),          # Pré-processamento
    ('classifier', RandomForestClassifier(n_estimators=100, max_depth=3, random_state=42))
])

# =========================
# Treinar modelo dentro de um run MLflow
# =========================
with mlflow.start_run():
    pipeline.fit(X_train, y_train)

    # Avaliação
    train_score = pipeline.score(X_train, y_train)
    test_score = pipeline.score(X_test, y_test)

    print(f"Train accuracy: {train_score:.3f}, Test accuracy: {test_score:.3f}")
