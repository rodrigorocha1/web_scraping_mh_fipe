from datetime import datetime
from typing import Dict, Any

import matplotlib.pyplot as plt
import numpy as np
from pandas import Series, DataFrame
from sklearn.model_selection import GridSearchCV
from sklearn.pipeline import Pipeline

from src.avaliador.avaliador import Avaliador


class AvaliadorXGboost(Avaliador):

    def __init__(self):
        pass

    def obter_dados_curva_validacao(self, pipeline: Pipeline, X_train: DataFrame, y_train: Series):
        pass

    def obter_resultados_modelo(self, pipeline: Pipeline, y_test: Series, y_pred: np.ndarray[Any, np.dtype[Any]]) -> \
            Dict[str, Any]:
        return {}

    def gerar_grafico_underfit_overfit(self, **kwargs):
        print(kwargs)

        treino_rmse = kwargs['train_rmse']
        validacao_rmse = kwargs['val_rmse']

        plt.plot(treino_rmse, label='Treino')
        plt.plot(validacao_rmse, label='Validação')
        plt.xlabel('Número de iterações')
        plt.ylabel('RMSE')
        plt.title('RMSE de treino e validação ao longo das iterações\ncom learning rate de 0.01')
        plt.legend()
        plt.savefig(
            f'fig/gerar_grafico_over_under/regressao_xgboost/gerar_grafico_underfit_overfit_regressao_xgboost_{datetime.now().strftime("%Y_%m_%d__%H_%M_%S")}.png'
        )

    def obter_resultado_grid_search(self, grid_search: GridSearchCV) -> Dict[str, Any]:
        best_param = {k: (float(v) if isinstance(v, (np.floating, np.integer)) else v)
                      for k, v in grid_search.best_params_.items()}

        # Converte o score negativo MSE para RMSE
        best_rmse = float(np.sqrt(np.abs(grid_search.best_score_)))

        resultado = {
            "melhores_hiperparametros": best_param,
            "melhor_rmse": best_rmse
        }

        return resultado
