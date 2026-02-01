from typing import Dict, Any, Final

from sklearn.ensemble import RandomForestRegressor

from src_machine_learning.estrategia_modelo.estrategia_modelo_sklearn import EstrategiaModeloSklearn
from src_machine_learning.utils.utils import carregar_dados_yaml_lista


class EstrategiaRegressaoRandomFlorest(EstrategiaModeloSklearn):
    PARAM_MODELO_REGRESSAO: Final[Dict[str, Any]] = \
        carregar_dados_yaml_lista(parametro_modelo='parametros_treinamento_simples')[0][
            'parametros']  # trazer do arquivo yaml
    PARAM_GRID: Final[Dict[str, Any]] = carregar_dados_yaml_lista(parametro_modelo='parametros_grid')[4][
        'parametros']  # trazer do arquivo yaml

    def __init__(self, polinomial: bool = False):
        super().__init__(
            modelo=('regressor', RandomForestRegressor(**self.PARAM_MODELO_REGRESSAO)),
            param_grid=self.PARAM_GRID,
            param_modelo_regressao=self.PARAM_MODELO_REGRESSAO,
            polinomial=polinomial

        )
