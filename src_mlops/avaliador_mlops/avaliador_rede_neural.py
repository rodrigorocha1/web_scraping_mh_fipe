from io import BytesIO
from typing import Dict, Any

import mlflow
import numpy as np
from PIL import Image
from matplotlib import pyplot as plt
from pandas import Series, DataFrame
from sklearn.metrics import mean_absolute_error, mean_squared_error, median_absolute_error, r2_score
from sklearn.model_selection import GridSearchCV, validation_curve
from sklearn.pipeline import Pipeline

from src_mlops.avaliador_mlops.avaliador import Avaliador


class AvaliadorRedeNeural(Avaliador):
    def __init__(self):
        self.__param_range = [(10,), (20,), (50,), (100,), (50, 50), (100, 50), (100, 100)]

    def obter_dados_curva_validacao(self, pipeline: Pipeline, X_train: DataFrame, y_train: Series) -> Dict[str, Any]:
        train_scores, val_scores = validation_curve(
            estimator=pipeline,
            X=X_train,
            y=y_train,
            param_name='regressor__regressor__hidden_layer_sizes',
            param_range=self.__param_range,
            cv=5,
            scoring='neg_root_mean_squared_error',
            n_jobs=-1
        )

        train_rmse = -train_scores.mean(axis=1)
        val_rmse = -val_scores.mean(axis=1)

        best_idx = np.argmin(val_rmse)
        best_hidden_layer = self.__param_range[best_idx]

        dados = {
            'train_rmse': train_rmse.tolist(),
            'val_rmse': val_rmse.tolist(),
            'best_idx': int(best_idx),
            'best_hidden_layer': list(best_hidden_layer)
        }
        return dados

    def obter_resultados_modelo(self, pipeline: Pipeline, y_test: Series, y_pred: np.ndarray[Any, np.dtype[Any]]) -> \
            Dict[str, Any]:
        assert pipeline is not None, "Pipeline não foi treinado"

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

        erro_percentual = np.abs(y_pred_arr - y_test_arr) / y_test_arr
        acc_10 = float(np.mean(erro_percentual <= 0.10))

        resultados: Dict[str, Any] = {
            "mae": mae,
            "rmse": rmse,
            "medae": medae,
            "smape": smape,
            "r2": r2,
            "bias": bias,
            "accuracy_erro_10_pct": acc_10,
            "n_amostras": len(y_test_arr),
            "preco_medio_real": float(np.mean(y_test_arr)),
            "preco_medio_previsto": float(np.mean(y_pred_arr)),
        }

        return resultados

    def gerar_grafico_underfit_overfit(self, dados: Dict[str, Any]):
        train_rmse = dados['train_rmse']
        val_rmse = dados['val_rmse']
        best_idx = dados['best_idx']
        best_hidden_layer = dados['best_hidden_layer']
        param_range = [str(h) for h in self.__param_range]

        # Cria a figura
        fig, ax = plt.subplots(figsize=(12, 10))

        # Plota curvas
        ax.plot(param_range, train_rmse, marker='o', label='RMSE Treino')
        ax.plot(param_range, val_rmse, marker='o', label='RMSE Validação')

        # Linha vertical no melhor hidden layer
        ax.axvline(best_idx, linestyle='--', label=f'Melhor Hidden Layer = {best_hidden_layer}')

        # Labels e título
        ax.set_xlabel('Arquitetura da Rede Neural')
        ax.set_ylabel('RMSE')
        ax.set_title('Rede Neural — Curva de Validação')
        ax.legend()
        ax.grid(True)
        fig.tight_layout()

        # Salvar no MLflow
        buf = BytesIO()
        fig.savefig(buf, format="png")
        buf.seek(0)
        img = Image.open(buf)
        mlflow.log_image(img, f"under_over_rede_neural.png")
        plt.close(fig)

    def obter_resultado_grid_search(self, grid_search: GridSearchCV) -> Dict[str, Any]:
        best_params = grid_search.best_params_

        # Função para converter tipos do NumPy para nativos do Python
        def to_native(val):
            if isinstance(val, (np.integer, np.int32, np.int64)):
                return int(val)
            if isinstance(val, (np.floating, np.float32, np.float64)):
                return float(val)
            if isinstance(val, np.ndarray):
                return val.tolist()
            return val

        # Converte todos os valores de best_params
        best_params_native = {k: to_native(v) for k, v in best_params.items()}

        # RMSE positivo
        best_rmse = float(-grid_search.best_score_)

        return {
            'best_params': best_params,
            'best_rmse': best_rmse
        }
