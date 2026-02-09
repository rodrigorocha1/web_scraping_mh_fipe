import mlflow
from sklearn.datasets import load_digits
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV, train_test_split


MLFLOW_URI = "http://172.25.0.5:5000"
mlflow.set_tracking_uri(MLFLOW_URI)
mlflow.set_registry_uri(MLFLOW_URI)

EXPERIMENT_NAME = "experimento_grid"
mlflow.set_experiment(EXPERIMENT_NAME)

# Ativa autologging do scikit-learn, mas se


# Load data
digits = load_digits()
X_train, X_test, y_train, y_test = train_test_split(
    digits.data, digits.target, test_size=0.2, random_state=42
)

# Enable autologging
mlflow.sklearn.autolog(max_tuning_runs=10)

# Define parameter grid
param_grid = {
    "n_estimators": [50, 100, 200],
    "max_depth": [5, 10, 15, None],
    "min_samples_split": [2, 5, 10],
}

with mlflow.start_run(run_name="RF Hyperparameter Tuning"):
    rf = RandomForestClassifier(random_state=42)
    grid_search = GridSearchCV(rf, param_grid, cv=5, scoring="accuracy", n_jobs=-1)
    grid_search.fit(X_train, y_train)

    best_score = grid_search.score(X_test, y_test)
    print(f"Best params: {grid_search.best_params_}")
    print(f"Best CV score: {grid_search.best_score_:.3f}")
    print(f"Test score: {best_score:.3f}")