import logging
from io import BytesIO
from typing import Dict, Any

import matplotlib.pyplot as plt
import mlflow
import numpy as np
import sklearn
from PIL import Image
from pandas import Series, DataFrame
from sklearn.base import BaseEstimator
from sklearn.metrics import mean_absolute_error, mean_squared_error, median_absolute_error, r2_score
from sklearn.model_selection import GridSearchCV, validation_curve

from src_mlops.avaliador_mlops.avaliador import Avaliador


class AvaliadorRegressaoLinear(Avaliador):
    def __init__(self):
        super().__init__()
        self._logger = logging.getLogger(self.__class__.__name__)
        self._param_range = np.logspace(-3, 2, 10)  # alpha

    def obter_dados_curva_validacao(self, pipeline: BaseEstimator, X_train: DataFrame, y_train: Series) -> Dict[
        str, Any]:
        regressor = pipeline.named_steps["regressor"]

        if not hasattr(regressor, "alpha"):
            return

        train_scores, val_scores = validation_curve(
            estimator=pipeline,
            X=X_train,
            y=y_train,
            param_name="regressor__alpha",
            param_range=self._param_range,
            cv=5,
            scoring="neg_root_mean_squared_error",
            n_jobs=-1
        )

        train_rmse = -train_scores.mean(axis=1)
        val_rmse = -val_scores.mean(axis=1)

        best_idx = np.argmin(val_rmse)

        return {
            "alpha_range": self._param_range.tolist(),
            "train_rmse": train_rmse.tolist(),
            "val_rmse": val_rmse.tolist(),
            "best_alpha": float(self._param_range[best_idx]),
            "best_rmse": float(val_rmse[best_idx]),
        }

    @staticmethod
    def __obter_nomes_features(pipeline: sklearn.pipeline.Pipeline) -> list[str]:
        preprocessor = pipeline.named_steps["preprocessor"]

        if hasattr(preprocessor, "get_feature_names_out"):
            return list(preprocessor.get_feature_names_out())

        raise ValueError("Preprocessor não expõe get_feature_names_out")

    def obter_resultados_modelo(self, pipeline: sklearn.pipeline.Pipeline, y_test: Series,
                                y_pred: np.ndarray[Any, np.dtype[Any]]) -> Dict[str, Any]:
        regressor = pipeline.named_steps["regressor"]

        y_test_arr = np.asarray(y_test)
        y_pred_arr = np.asarray(y_pred)

        mae = mean_absolute_error(y_test_arr, y_pred_arr)
        rmse = np.sqrt(mean_squared_error(y_test_arr, y_pred_arr))
        medae = median_absolute_error(y_test_arr, y_pred_arr)
        r2 = r2_score(y_test_arr, y_pred_arr)

        smape = np.mean(
            2.0 * np.abs(y_pred_arr - y_test_arr)
            / (np.abs(y_test_arr) + np.abs(y_pred_arr))
        ) * 100

        bias = float(np.mean(y_pred_arr - y_test_arr))
        erro_pct = np.abs(y_pred_arr - y_test_arr) / y_test_arr
        acc_10 = float(np.mean(erro_pct <= 0.10))
        nomes_features = self.__obter_nomes_features(pipeline)

        coeficientes = {
            nome: float(coef)
            for nome, coef in zip(nomes_features, regressor.coef_)
        }

        return {
            "mae": mae,
            "rmse": float(rmse),
            "medae": medae,
            "smape": float(smape),
            "r2": r2,
            "bias": bias,
            "accuracy_erro_10_pct": acc_10,

            "preco_medio_real": float(np.mean(y_test_arr)),
            "preco_medio_previsto": float(np.mean(y_pred_arr)),
            "n_amostras": len(y_test_arr),

            # Interpretabilidade
            "intercepto": float(regressor.intercept_),
            "coeficientes": coeficientes,
        }

    def gerar_grafico_underfit_overfit(self, dados: Dict[str, Any]):
        if not dados:
            self._logger.warning("Sem dados para gerar gráfico")
            return

        alpha = dados["alpha_range"]
        train_rmse = dados["train_rmse"]
        val_rmse = dados["val_rmse"]

        # Cria figura e eixo
        fig, ax = plt.subplots(figsize=(12, 7))

        # Plota curvas em escala logarítmica
        ax.semilogx(alpha, train_rmse, marker="o", label="Treino")
        ax.semilogx(alpha, val_rmse, marker="o", label="Validação")

        # Labels e título
        ax.set_xlabel("alpha (Regularização)")
        ax.set_ylabel("RMSE")
        ax.set_title("Regressão Linear Regularizada — Bias vs Variance")
        ax.legend()
        ax.grid(True)
        fig.tight_layout()

        # Salvar no MLflow
        buf = BytesIO()
        fig.savefig(buf, format="png")
        buf.seek(0)
        img = Image.open(buf)
        mlflow.log_image(img, f"under_over_linear.png")

        plt.close(fig)

    def obter_resultado_grid_search(self, grid_search: GridSearchCV) -> Dict[str, Any]:
        assert grid_search is not None, "GridSearchCV não foi executado"

        best_params = grid_search.best_params_
        best_estimator = grid_search.best_estimator_
        best_score = grid_search.best_score_

        rmse = abs(float(best_score))

        def to_native(valor: Any) -> Any:
            if isinstance(valor, (np.integer,)):
                return int(valor)
            if isinstance(valor, (np.floating,)):
                return float(valor)
            if isinstance(valor, np.ndarray):
                return valor.tolist()
            return valor

        best_params_native = {
            chave: to_native(valor)
            for chave, valor in best_params.items()
        }

        resultado: Dict[str, Any] = {
            "best_params": best_params_native,
            "best_score": float(best_score),
            "rmse": rmse,
            "best_estimator": best_estimator.__class__.__name__,
        }

        return resultado
