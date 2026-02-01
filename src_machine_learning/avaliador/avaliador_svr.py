from datetime import datetime
from typing import Dict, Any

import matplotlib.pyplot as plt
import numpy as np
from pandas import Series, DataFrame
from sklearn.metrics import mean_absolute_error, mean_squared_error, median_absolute_error, r2_score
from sklearn.model_selection import GridSearchCV, validation_curve
from sklearn.pipeline import Pipeline

from src.avaliador.avaliador import Avaliador


class AvaliadorSVR(Avaliador):
    def __init__(self):
        self.__param_range = np.logspace(-2, 3, 10)

    def obter_dados_curva_validacao(self, pipeline: Pipeline, X_train: DataFrame, y_train: Series) -> Dict[str, Any]:
        train_scores, val_scores = validation_curve(
            estimator=pipeline,
            X=X_train,
            y=y_train,
            param_name='regressor__C',
            param_range=self.__param_range,
            cv=5,
            scoring='neg_root_mean_squared_error',
            n_jobs=-1
        )

        train_rmse = -train_scores.mean(axis=1)
        val_rmse = -val_scores.mean(axis=1)

        best_idx = np.argmin(val_rmse)
        best_C = self.__param_range[best_idx]

        dados = {
            'train_rmse': train_rmse.tolist(),
            'val_rmse': val_rmse.tolist(),
            'best_idx': int(best_idx),
            'best_C': float(best_C)
        }
        return dados

    def obter_resultados_modelo(self, pipeline: Pipeline, y_test: Series, y_pred:  np.ndarray[Any, np.dtype[Any]]) -> Dict[str, Any]:
        assert pipeline is not None, "Pipeline não foi treinado"

        y_test_arr = np.asarray(y_test)
        y_pred_arr = np.asarray(y_pred)

        mae = mean_absolute_error(y_test_arr, y_pred_arr)
        rmse = np.sqrt(mean_squared_error(y_test_arr, y_pred_arr))
        medae = median_absolute_error(y_test_arr, y_pred_arr)
        r2 = r2_score(y_test_arr, y_pred_arr)

        # SMAPE (robusto para imóveis)
        smape = np.mean(
            2.0 * np.abs(y_pred_arr - y_test_arr)
            / (np.abs(y_test_arr) + np.abs(y_pred_arr))
        ) * 100

        bias = float(np.mean(y_pred_arr - y_test_arr))

        erro_percentual = np.abs(y_pred_arr - y_test_arr) / y_test_arr
        acc_10 = float(np.mean(erro_percentual <= 0.10))

        resultados: Dict[str, Any] = {
            "mae": mae,
            "rmse": float(rmse),
            "medae": medae,
            "smape": float(smape),
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
        best_C = dados['best_C']
        param_range = self.__param_range

        plt.figure(figsize=(10, 6))
        plt.plot(param_range, train_rmse, marker='o', label='RMSE Treino')
        plt.plot(param_range, val_rmse, marker='o', label='RMSE Validação')
        plt.axvline(best_C, linestyle='--', label=f'Melhor C = {best_C:.3f}')
        plt.xscale('log')
        plt.xlabel('C')
        plt.ylabel('RMSE')
        plt.title('SVR — Curva de Validação')
        plt.legend()
        plt.grid(True)
        plt.tight_layout()
        plt.savefig(
            f'fig/gerar_grafico_over_under/{dados["nome_modelo"]}/gerar_grafico_underfit_overfit_svr_{datetime.now().strftime("%Y_%m_%d__%H_%M_%S")}.png'
        )

    def obter_resultado_grid_search(self, grid_search: GridSearchCV) ->  Dict[str, Any]:
        def to_native(val):
            if isinstance(val, (np.integer, np.int32, np.int64)):
                return int(val)
            if isinstance(val, (np.floating, np.float32, np.float64)):
                return float(val)
            if isinstance(val, np.ndarray):
                return val.tolist()
            if isinstance(val, list):
                return [to_native(v) for v in val]
            if isinstance(val, dict):
                return {k: to_native(v) for k, v in val.items()}
            return val

        best_params_native = {k: to_native(v) for k, v in grid_search.best_params_.items()}
        best_rmse = float(-grid_search.best_score_)

        # Extrair alguns valores detalhados do GridSearch

        return {
            'best_params': grid_search.best_params_,
            'best_rmse': best_rmse

        }
