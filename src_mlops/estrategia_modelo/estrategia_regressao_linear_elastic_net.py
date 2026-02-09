from typing import Final, Dict, Any

from sklearn.linear_model import ElasticNet
from sklearn.preprocessing import PolynomialFeatures

from src_machine_learning.estrategia_modelo.estrategia_modelo_sklearn import EstrategiaModeloSklearn
from src_machine_learning.utils.utils import carregar_dados_yaml_lista


class EstrategiaRegressaoElasticNet(EstrategiaModeloSklearn):
    PARAM_MODELO_REGRESSAO: Final[Dict[str, Any]] = \
        carregar_dados_yaml_lista(parametro_modelo='parametros_treinamento_simples')[8][
            'parametros']  # trazer do arquivo yaml
    PARAM_GRID: Final[Dict[str, Any]] = carregar_dados_yaml_lista(parametro_modelo='parametros_grid')[7][
        'parametros']  # trazer do arquivo yaml

    PARAM_POLINOMIAL: Final[Dict[str, Any]] = \
        carregar_dados_yaml_lista(parametro_modelo='param_grid_regressao_polinomial')[1][
            'parametros']
    modelo = ElasticNet(**PARAM_MODELO_REGRESSAO)

    def __init__(self, polinomial: bool = False, opcao: int = 1):
        self.__opcao = opcao
        super().__init__(
            param_modelo_regressao=self.PARAM_MODELO_REGRESSAO,
            modelo=('regressor', self.modelo) if self.__opcao == 1 else ('regressor', ElasticNet()),
            param_grid=self.PARAM_POLINOMIAL if polinomial else self.PARAM_GRID,
            modelo_polinomial=("poly", PolynomialFeatures()),
            polinomial=polinomial
        )
