import logging
from io import BytesIO
from typing import Dict, Any

import matplotlib.pyplot as plt
import mlflow
import numpy as np
import sklearn
from PIL import Image
from pandas import Series, DataFrame
from sklearn.metrics import (
    mean_absolute_error,
    mean_squared_error,
    median_absolute_error,
    r2_score,
)
from sklearn.model_selection import GridSearchCV, validation_curve
from sklearn.pipeline import Pipeline

from src_machine_learning.avaliador.avaliador import Avaliador


class AvaliadorRegressaoLinearRegularizada(Avaliador):
    """
    Avaliador para Ridge, Lasso e ElasticNet
    """

    def __init__(self):
        super().__init__()
        self._logger = logging.getLogger(self.__class__.__name__)
        self._alpha_range = np.logspace(-3, 2, 10)

    # ==========================================================
    # Curva de validação (Bias vs Variance)
    # ==========================================================
    def obter_dados_curva_validacao(
            self,
            pipeline: Pipeline,
            X_train: DataFrame,
            y_train: Series,
    ) -> Dict[str, Any]:

        regressor = pipeline.named_steps["regressor"]

        if not hasattr(regressor, "alpha"):
            raise ValueError(
                "Este avaliador suporta apenas Ridge, Lasso e ElasticNet"
            )

        train_scores, val_scores = validation_curve(
            estimator=pipeline,
            X=X_train,
            y=y_train,
            param_name="regressor__alpha",
            param_range=self._alpha_range,
            cv=5,
            scoring="neg_root_mean_squared_error",
            n_jobs=-1,
        )

        train_rmse = -train_scores.mean(axis=1)
        val_rmse = -val_scores.mean(axis=1)

        best_idx = int(np.argmin(val_rmse))

        return {
            "alpha_range": self._alpha_range.tolist(),
            "train_rmse": train_rmse.tolist(),
            "val_rmse": val_rmse.tolist(),
            "best_alpha": float(self._alpha_range[best_idx]),
            "best_rmse": float(val_rmse[best_idx]),
        }

    @staticmethod
    def __obter_nomes_features(pipeline: sklearn.pipeline.Pipeline) -> list[str]:
        preprocessor = pipeline.named_steps["preprocessor"]

        if hasattr(preprocessor, "get_feature_names_out"):
            return list(preprocessor.get_feature_names_out())

        raise ValueError("Preprocessor não expõe get_feature_names_out")

    def obter_resultados_modelo(
            self,
            pipeline: Pipeline,
            y_test: Series,
            y_pred: np.ndarray,
    ) -> Dict[str, Any]:

        regressor = pipeline.named_steps["regressor"]

        y_test_arr = np.asarray(y_test)
        y_pred_arr = np.asarray(y_pred)

        mae = mean_absolute_error(y_test_arr, y_pred_arr)
        rmse = np.sqrt(mean_squared_error(y_test_arr, y_pred_arr))
        medae = median_absolute_error(y_test_arr, y_pred_arr)
        r2 = r2_score(y_test_arr, y_pred_arr)

        # Evita divisão por zero no SMAPE
        denom = np.abs(y_test_arr) + np.abs(y_pred_arr)
        smape = np.mean(
            np.where(denom == 0, 0.0, 2.0 * np.abs(y_pred_arr - y_test_arr) / denom)
        ) * 100

        bias = float(np.mean(y_pred_arr - y_test_arr))
        erro_pct = np.abs(y_pred_arr - y_test_arr) / np.clip(
            np.abs(y_test_arr), 1e-8, None
        )
        acc_10 = float(np.mean(erro_pct <= 0.10))

        nomes_features = self.__obter_nomes_features(pipeline)
        coeficientes = {
            nome: float(coef)
            for nome, coef in zip(nomes_features, regressor.coef_)
        }
        resultado = {
            # Métricas
            "mae": float(mae),
            "rmse": float(rmse),
            "medae": float(medae),
            "smape": float(smape),
            "r2": float(r2),
            "bias": bias,
            "accuracy_erro_10_pct": acc_10,

            # Estatísticas
            "preco_medio_real": float(np.mean(y_test_arr)),
            "preco_medio_previsto": float(np.mean(y_pred_arr)),
            "n_amostras": int(len(y_test_arr)),

            # Modelo
            "modelo": regressor.__class__.__name__,
            "alpha": float(regressor.alpha),
            "intercepto": float(regressor.intercept_),
            "coeficientes": coeficientes,
        }

        if hasattr(regressor, "l1_ratio"):
            resultado["l1_ratio"] = float(regressor.l1_ratio)

        return resultado

    def gerar_grafico_underfit_overfit(self, dados: Dict[str, Any]) -> None:
        if not dados:
            self._logger.warning("Sem dados para gerar gráfico")
            return

        alpha_range = dados["alpha_range"]
        train_rmse = dados["train_rmse"]
        val_rmse = dados["val_rmse"]
        nome_modelo = dados.get("nome_modelo", "modelo_desconhecido")

        # Criar figura e eixo
        fig, ax = plt.subplots(figsize=(12, 7))

        # Plota curvas em escala logarítmica
        ax.semilogx(alpha_range, train_rmse, marker="o", label="Treino (Viés)")
        ax.semilogx(alpha_range, val_rmse, marker="o", label="Validação (Generalização)")

        # Labels e título
        ax.set_xlabel("alpha (Regularização)")
        ax.set_ylabel("RMSE")
        ax.set_title(f"Regressão Linear Regularizada - {nome_modelo} — Bias vs Variance")
        ax.legend()
        ax.grid(True)
        fig.tight_layout()

        # Salvar gráfico direto no MLflow
        buf = BytesIO()
        fig.savefig(buf, format="png")
        buf.seek(0)
        img = Image.open(buf)
        mlflow.log_image(img, f"under_over_{nome_modelo}.png")

        plt.close(fig)

    def obter_resultado_grid_search(
            self, grid_search: GridSearchCV
    ) -> Dict[str, Any]:

        assert grid_search is not None, "GridSearchCV não foi executado"

        def to_native(v: Any) -> Any:
            if isinstance(v, (np.integer,)):
                return int(v)
            if isinstance(v, (np.floating,)):
                return float(v)
            if isinstance(v, np.ndarray):
                return v.tolist()
            return v

        return {
            "best_params": {
                k: to_native(v)
                for k, v in grid_search.best_params_.items()
            },
            "best_score": float(grid_search.best_score_),
            "rmse": abs(float(grid_search.best_score_)),
            "best_estimator": grid_search.best_estimator_.__class__.__name__,
        }
