from abc import ABC, abstractmethod
from typing import Dict, Any

import numpy as np
import sklearn
from pandas import DataFrame, Series
from sklearn.base import BaseEstimator
from sklearn.model_selection import GridSearchCV


class Avaliador(ABC):

    @abstractmethod
    def obter_dados_curva_validacao(self, pipeline: BaseEstimator, X_train: DataFrame, y_train: Series) -> \
            Dict[str, Any]:
        pass

    @abstractmethod
    def obter_resultados_modelo(self, pipeline: sklearn.pipeline.Pipeline, y_test: Series,
                                y_pred: np.ndarray[Any, np.dtype[Any]]) -> \
            Dict[str, Any]:
        pass

    @abstractmethod
    def gerar_grafico_underfit_overfit(self, metricas: Dict[str, Any]):
        pass

    @abstractmethod
    def obter_resultado_grid_search(self, grid_search: GridSearchCV) -> Dict[str, Any]:
        pass
