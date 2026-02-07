from typing import Dict, Any, Final

import numpy as np
import xgboost as xgb
from pandas import DataFrame, Series
from sklearn.model_selection import GridSearchCV
from xgboost import Booster

from src_machine_learning.estrategia_modelo.estrategia_modelo import EstrategiaModelo, ResultadoBuscaT
from src_machine_learning.utils.utils import carregar_dados_yaml_lista


class EstrategiaModeloXgboost(EstrategiaModelo):
    PARAM_MODELO_REGRESSAO: Final[Dict[str, Any]] = (
        carregar_dados_yaml_lista(
            parametro_modelo='parametros_treinamento_simples'
        )[1]['parametros']
    )

    PARAM_GRID: Final[Dict[str, Any]] = (
        carregar_dados_yaml_lista(
            parametro_modelo='parametros_grid'
        )[1]['parametros']
    )

    modelo = xgb.XGBRFRegressor(**PARAM_MODELO_REGRESSAO)

    def __init__(self):
        self.__modelo_treinado: Booster | None = None
        super().__init__()

    @staticmethod
    def _converter_categorias(df: DataFrame) -> DataFrame:
        df = df.copy()
        for col in df.select_dtypes(include='category').columns:
            df[col] = df[col].cat.codes
        return df

    def treinar_modelo(self, **kwargs) -> None:
        x_train: DataFrame = kwargs['x_train']
        y_train: Series = kwargs['y_train']

        x_train = self._converter_categorias(x_train)

        dtrain = xgb.DMatrix(
            data=x_train,
            label=y_train,
            enable_categorical=True
        )
        evals_result = {}
        self.__modelo_treinado = xgb.train(
            params=self.PARAM_MODELO_REGRESSAO,
            dtrain=dtrain,
            num_boost_round=100,
            evals_result=evals_result
        )
        self._resultados_modelo = evals_result

    def predizer_modelo(self, **kwargs) -> np.ndarray:
        x_test: DataFrame = kwargs['x_test']

        x_test = self._converter_categorias(x_test)

        dtest = xgb.DMatrix(
            data=x_test,
            enable_categorical=True
        )

        return self.__modelo_treinado.predict(dtest)

    def realizar_grid_search(self, x: DataFrame, y: Series) -> ResultadoBuscaT:
        x = self._converter_categorias(x)

        estimator = xgb.XGBRegressor(
            enable_categorical=True,
            objective=self.PARAM_MODELO_REGRESSAO.get('objective', 'reg:squarederror')
        )

        grid = GridSearchCV(
            estimator=estimator,
            param_grid=self.PARAM_GRID,
            scoring='neg_mean_squared_error',
            cv=5,
            verbose=1
        )

        grid.fit(x, y)
        return grid

    def realizar_validacao_cruzada(
            self,
            x: DataFrame,
            y: Series,
            iteracao: int
    ) -> Dict[str, Any]:
        x = self._converter_categorias(x)

        dtrain = xgb.DMatrix(
            data=x,
            label=y,
            enable_categorical=True
        )

        cv_resultados = xgb.cv(
            params=self.PARAM_MODELO_REGRESSAO,
            dtrain=dtrain,
            nfold=5,
            num_boost_round=1000,
            early_stopping_rounds=10,
            metrics='rmse',
            seed=4789,
            as_pandas=False
        )

        return cv_resultados
