from copy import deepcopy
from datetime import datetime
from typing import Dict, Any, Final

import numpy as np
import pandas as pd
from pandas import DataFrame, Series
from sklearn.base import clone
from sklearn.model_selection import GridSearchCV, cross_validate, KFold, cross_val_predict
from sklearn.pipeline import Pipeline

from src_machine_learning.config.variaveis import EtapaRegressao, PassoPipelineSklearn
from src_machine_learning.estrategia_modelo.estrategia_modelo import EstrategiaModelo


class EstrategiaModeloSklearn(EstrategiaModelo[PassoPipelineSklearn, GridSearchCV]):

    def __init__(
            self,
            param_modelo_regressao: Dict[str, Any],
            modelo: EtapaRegressao,
            param_grid: Dict[str, Any],
            polinomial: bool = False,
            modelo_polinomial: EtapaRegressao = None
    ):
        super().__init__(polinomial=polinomial)
        self.__params = param_modelo_regressao
        self.__modelo = modelo
        self.__modelo_polinomial = modelo_polinomial

        self.__PARAM_GRID: Final[Dict[str, Any]] = param_grid

    def treinar_modelo(self, **kwargs) -> None:
        x: DataFrame = kwargs['x']
        y: Series = kwargs['y']

        assert self._pipeline is not None


        if self._polinomial:
            self._pipeline.append(self.__modelo_polinomial)
        self._pipeline.append(self.__modelo)
        self._dados_treinamento = Pipeline(steps=self._pipeline)

        self._dados_treinamento.fit(x, y)

    def predizer_modelo(self, x_test: pd.DataFrame) -> np.ndarray[Any]:
        assert self._dados_treinamento is not None, "O modelo precisa ser treinado antes de predizer!"
        return self._dados_treinamento.predict(x_test)

    def realizar_grid_search(
            self, x: DataFrame, y: Series
    ) -> GridSearchCV:
        assert self._pipeline is not None

        if self._polinomial:
            self._pipeline.append(self.__modelo_polinomial)
        self._pipeline.append(self.__modelo)
        pipeline = Pipeline(self._pipeline)


        grid_search = GridSearchCV(
            estimator=pipeline,
            param_grid=self.__PARAM_GRID,
            scoring='neg_root_mean_squared_error',
            cv=5,
            n_jobs=-1,
            verbose=1,
            return_train_score=True,

        )

        grid_search.fit(x, y)
        return grid_search

    def realizar_validacao_cruzada(
            self,
            x: np.ndarray[Any],
            y: np.ndarray[Any],
            iteracao: int
    ) -> Dict[str, Any]:
        assert self._pipeline is not None
        passos = deepcopy(self._pipeline)

        if not any(nome == 'regressor' for nome, _ in passos):
            passos.append(self.__modelo)

        pipeline = Pipeline(passos)

        kfold = KFold(n_splits=10, shuffle=True, random_state=iteracao)

        scoring = ['r2', 'neg_mean_squared_error', 'neg_mean_absolute_error']

        scores = cross_validate(
            pipeline,
            x,
            y,
            cv=kfold,
            scoring=scoring,
            return_train_score=True,
            error_score='raise'
        )

        results_dict = {
            "test_r2": scores['test_r2'].tolist(),
            "test_mse": (-scores['test_neg_mean_squared_error']).tolist(),
            "test_mae": (-scores['test_neg_mean_absolute_error']).tolist(),
            "train_r2": scores['train_r2'].tolist(),
            "train_mse": (-scores['train_neg_mean_squared_error']).tolist(),
            "train_mae": (-scores['train_neg_mean_absolute_error']).tolist(),
            "fit_time": scores['fit_time'].tolist(),
            "score_time": scores['score_time'].tolist(),
        }

        # 🔹 RMSE por fold
        results_dict["test_rmse"] = [float(np.sqrt(mse)) for mse in results_dict["test_mse"]]
        results_dict["train_rmse"] = [float(np.sqrt(mse)) for mse in results_dict["train_mse"]]

        mean_scores = {
            "mean_test_r2": float(np.mean(results_dict["test_r2"])),
            "mean_test_mse": float(np.mean(results_dict["test_mse"])),
            "mean_test_mae": float(np.mean(results_dict["test_mae"])),
            "mean_test_rmse": float(np.mean(results_dict["test_rmse"])),
            "mean_train_r2": float(np.mean(results_dict["train_r2"])),
            "mean_train_mse": float(np.mean(results_dict["train_mse"])),
            "mean_train_mae": float(np.mean(results_dict["train_mae"])),
            "mean_train_rmse": float(np.mean(results_dict["train_rmse"])),
            "mean_fit_time": float(np.mean(results_dict["fit_time"])),
            "mean_score_time": float(np.mean(results_dict["score_time"]))
        }

        # 🔹 Predição out-of-fold (resíduos globais)
        y_pred = cross_val_predict(clone(pipeline), x, y, cv=kfold, n_jobs=-1)
        residuos_totais = y - y_pred

        rmse_folds = np.sqrt(-scores['test_neg_mean_squared_error']).tolist()

        return {
            "results_dict": results_dict,
            "mean_scores": mean_scores,
            "residuos_totais": residuos_totais.tolist(),
            "iteracao": iteracao,
            "rmse_folds": rmse_folds,
            "data_coleta": datetime.now().strftime("%d/%m/%Y %H:%M:%S")
        }
