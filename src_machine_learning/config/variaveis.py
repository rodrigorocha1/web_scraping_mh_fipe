from typing import Tuple
from typing import TypeAlias, List, Union

import xgboost as xgb
from pandas import DataFrame, Series
from sklearn.base import RegressorMixin
from sklearn.compose import ColumnTransformer
from sklearn.model_selection import GridSearchCV
from sklearn.preprocessing import FunctionTransformer
from xgboost import DMatrix

EtapaTranformacao: TypeAlias = Tuple[str, Union[
    FunctionTransformer,
    ColumnTransformer
]]

TransformacaoXgBoost = Tuple[DMatrix, DMatrix]

EtapaRegressao: TypeAlias = Tuple[str, RegressorMixin]

PassoPipelineSklearn: TypeAlias = List[Union[
    EtapaTranformacao,
    EtapaRegressao
]]

ModeloXGB = xgb.Booster
ResultadoBuscaXGB = GridSearchCV

SeparacaoTreinoTeste = Tuple[DataFrame, DataFrame, Series, Series]
PassoPipelineXgboost = Tuple[DataFrame, DataFrame, Series, Series]
