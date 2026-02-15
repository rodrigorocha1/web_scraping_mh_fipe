from sklearn.datasets import load_diabetes
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split

import mlflow
import mlflow.sklearn

MLFLOW_URI = "http://localhost:5000"
mlflow.set_tracking_uri(MLFLOW_URI)

print("Tracking URI:", mlflow.get_tracking_uri())

X, y = load_diabetes(return_X_y=True)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

with mlflow.start_run() as run:
    model = LinearRegression()
    model.fit(X_train, y_train)

    preds = model.predict(X_test)
    rmse = mean_squared_error(y_test, preds,)

    mlflow.log_metric("rmse", rmse)

    mlflow.sklearn.log_model(
        model,
        artifact_path="model",
        registered_model_name="modelo-teste-api"
    )

    run_id = run.info.run_id

print("Run ID:", run_id)
