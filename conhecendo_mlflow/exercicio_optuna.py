import mlflow
import optuna
from sklearn.datasets import load_breast_cancer
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.model_selection import train_test_split


MLFLOW_URI = "http://172.25.0.5:5000"
mlflow.set_tracking_uri(MLFLOW_URI)
mlflow.set_registry_uri(MLFLOW_URI)

EXPERIMENT_NAME = "experimento_hiperparametro"
mlflow.set_experiment(EXPERIMENT_NAME)


cancer = load_breast_cancer()
X_train, X_test, y_train, y_test = train_test_split(
    cancer.data, cancer.target, test_size=0.2, random_state=42
)

mlflow.sklearn.autolog()


def objective(trial):
    params = {
        "n_estimators": trial.suggest_int("n_estimators", 50, 200),
        "max_depth": trial.suggest_int("max_depth", 3, 10),
        "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.3),
    }

    with mlflow.start_run(nested=True):
        model = GradientBoostingClassifier(**params, random_state=42)
        model.fit(X_train, y_train)
        accuracy = model.score(X_test, y_test)
        return accuracy


with mlflow.start_run(run_name='exercicio_optuna'):
    study = optuna.create_study(direction="maximize")
    study.optimize(objective, n_trials=20)

    mlflow.log_params({f"best_{k}": v for k, v in study.best_params.items()})
    mlflow.log_metric("best_accuracy", study.best_value)