import mlflow
from sklearn.datasets import load_wine
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split

MLFLOW_URI = "http://172.25.0.5:5000"
mlflow.set_tracking_uri(MLFLOW_URI)
mlflow.set_registry_uri(MLFLOW_URI)

EXPERIMENT_NAME = "experimento_grid"
mlflow.set_experiment(EXPERIMENT_NAME)
# Enable autologging
mlflow.sklearn.autolog()

# Load and prepare data
wine = load_wine()
X_train, X_test, y_train, y_test = train_test_split(
    wine.data, wine.target, test_size=0.2, random_state=42
)

# Train model - MLflow automatically logs everything!
with mlflow.start_run(run_name='teste_um'):
    model = RandomForestClassifier(n_estimators=100, max_depth=3, random_state=42)
    model.fit(X_train, y_train)

    # Evaluation metrics are automatically captured
    train_score = model.score(X_train, y_train)
    test_score = model.score(X_test, y_test)

    print(f"Train accuracy: {train_score:.3f}, Test accuracy: {test_score:.3f}")
