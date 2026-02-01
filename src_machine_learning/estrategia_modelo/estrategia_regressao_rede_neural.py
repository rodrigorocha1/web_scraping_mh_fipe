from typing import Dict, Final, Any

from sklearn.compose import TransformedTargetRegressor
from sklearn.neural_network import MLPRegressor
from sklearn.preprocessing import StandardScaler

from src_machine_learning.estrategia_modelo.estrategia_modelo_sklearn import EstrategiaModeloSklearn
from src_machine_learning.utils.utils import carregar_dados_yaml_lista


class EstrategiaRegressaoRedeNeural(EstrategiaModeloSklearn):
    PARAM_MODELO_REGRESSAO: Final[Dict[str, Any]] = \
    carregar_dados_yaml_lista(parametro_modelo='parametros_treinamento_simples')[3][
        'parametros']  # trazer do arquivo yaml
    PARAM_GRID: Final[Dict[str, Any]] = carregar_dados_yaml_lista(parametro_modelo='parametros_grid')[2][
        'parametros']  # trazer do arquivo yaml

    def __init__(self, ):
        super().__init__(
            param_modelo_regressao=self.PARAM_MODELO_REGRESSAO,
            modelo=('regressor', TransformedTargetRegressor(
                regressor=MLPRegressor(
                    **self.PARAM_MODELO_REGRESSAO
                ),
                transformer=StandardScaler()
            )),
            param_grid=self.PARAM_GRID
        )
