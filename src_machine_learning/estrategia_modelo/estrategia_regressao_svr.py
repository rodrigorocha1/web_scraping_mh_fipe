from typing import Dict, Final, Any

from sklearn.svm import SVR

from src_machine_learning.estrategia_modelo.estrategia_modelo_sklearn import EstrategiaModeloSklearn
from src_machine_learning.utils.utils import carregar_dados_yaml_lista


class EstrategiaRegressaoSVR(EstrategiaModeloSklearn):
    PARAM_MODELO_REGRESSAO: Final[Dict[str, Any]] = \
        carregar_dados_yaml_lista(parametro_modelo='parametros_treinamento_simples')[2][
            'parametros']  # trazer do arquivo yaml
    PARAM_GRID: Final[Dict[str, Any]] = carregar_dados_yaml_lista(parametro_modelo='parametros_grid')[3]['parametros']

    def __init__(self, ):
        print(self.PARAM_MODELO_REGRESSAO)
        super().__init__(
            param_modelo_regressao=self.PARAM_MODELO_REGRESSAO,
            modelo=('regressor', SVR(
                **self.PARAM_MODELO_REGRESSAO
            )),

            param_grid=self.PARAM_GRID
        )
