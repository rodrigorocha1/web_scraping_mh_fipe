# from typing import Dict, Any, Final
#
# import xgboost as xgb
#
# from src.estrategia.estrategia_modelo_xgboost import EstrategiaModeloXgboost
# from src.utils.utils import carregar_dados_yaml_lista
#
#
# class EstrategiaXgboost(EstrategiaModeloXgboost):
#     PARAM_MODELO_REGRESSAO: Final[Dict[str, Any]] = (
#         carregar_dados_yaml_lista(
#             parametro_modelo='parametros_treinamento_simples'
#         )[1]['parametros']
#     )
#
#     PARAM_GRID: Final[Dict[str, Any]] = (
#         carregar_dados_yaml_lista(
#             parametro_modelo='parametros_grid'
#         )[1]['parametros']
#     )
#
#     modelo = xgb.XGBRFRegressor(**PARAM_MODELO_REGRESSAO)
#
#     def __init__(self):
#         super().__init__()
