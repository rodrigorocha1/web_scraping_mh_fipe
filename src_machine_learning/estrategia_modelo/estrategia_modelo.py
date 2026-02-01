from abc import ABC, abstractmethod
from typing import Dict, Any, TypeVar, Generic, Optional, List, Tuple

import numpy as np
from pandas import DataFrame, Series
from sklearn.pipeline import Pipeline

PassosPipelineModelo = TypeVar("PassosPipelineModelo")
ResultadoBuscaT = TypeVar("ResultadoBuscaT")


class EstrategiaModelo(ABC, Generic[PassosPipelineModelo, ResultadoBuscaT]):

    def __init__(self) -> None:
        self._resultados_modelo: Dict[str, Any]  = {}
        self._pipeline : List = []

        self._dados_treinamento: Optional[Pipeline] = None

    @property
    def dados_treinamento(self):
        return self._dados_treinamento

    @property
    def pipeline(self) -> List[PassosPipelineModelo] :
        return self._pipeline

    @pipeline.setter
    def pipeline(self, pipeline):
        self._pipeline = pipeline

    @abstractmethod
    def treinar_modelo(self, **kwargs) -> None:
        pass

    @abstractmethod
    def predizer_modelo(self, x_test: DataFrame) -> np.ndarray[Any]:
        pass

    @abstractmethod
    def realizar_grid_search(
            self,
            x: DataFrame,
            y: Series,
    ) -> ResultadoBuscaT:
        pass

    @abstractmethod
    def realizar_validacao_cruzada(
            self,
            x: np.ndarray[Any],
            y: np.ndarray[Any],
            iteracao: int
    ) -> Dict[str, Any]:
        pass
