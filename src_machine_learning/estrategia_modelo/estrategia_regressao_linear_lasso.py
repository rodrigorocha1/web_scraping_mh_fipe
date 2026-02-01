from typing import Final, Dict, Any

from sklearn.linear_model import Ridge, Lasso

from src_machine_learning.estrategia_modelo.estrategia_modelo_sklearn import EstrategiaModeloSklearn
from src_machine_learning.utils.utils import carregar_dados_yaml_lista


class EstrategiaRegressaoLinearLasso(EstrategiaModeloSklearn):
    PARAM_MODELO_REGRESSAO: Final[Dict[str, Any]] = \
        carregar_dados_yaml_lista(parametro_modelo='parametros_treinamento_simples')[8][
            'parametros']  # trazer do arquivo yaml
    PARAM_GRID: Final[Dict[str, Any]] = carregar_dados_yaml_lista(parametro_modelo='parametros_grid')[8][
        'parametros']  # trazer do arquivo yaml
    modelo = Lasso(**PARAM_MODELO_REGRESSAO)


    def __init__(self):
        super().__init__(
            param_modelo_regressao=self.PARAM_MODELO_REGRESSAO,
            modelo=('regressor', self.modelo),
            param_grid=self.PARAM_GRID
        )