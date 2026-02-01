from dataclasses import dataclass
from typing import Generic, TypeVar

from pandas import DataFrame, Series
from sklearn.model_selection import GridSearchCV

from src.config.variaveis import PassoPipelineSklearn
from src.preprocessador.preprocessador_sklearn import PreprocessadorSklearn

T = TypeVar("T")


@dataclass
class ResultadoPreprocessamento(Generic[T]):
    x_train: DataFrame
    x_test: DataFrame
    y_train: Series
    y_test: Series
    pipeline: T


from datetime import datetime
from typing import Final, List, TypeVar, Generic

from pandas import DataFrame, Series

from src.avaliador.avaliador import Avaliador
from src.estrategia.estrategia_modelo import EstrategiaModelo
from src.preprocessador.preprocessador import Preprocessador
from src.utils.utils import salvar_json

PassoPipelinePreprocessador = TypeVar("PassoPipelinePreprocessador")
ResultadoBuscaT = TypeVar("ResultadoBuscaT")


class PipelineCompletaMachineLearning(
    Generic[PassoPipelinePreprocessador, ResultadoBuscaT]
):
    def __init__(
            self,
            preprocessamento_modelo: Preprocessador[PassoPipelinePreprocessador],
            treinamento_modelo: EstrategiaModelo[
                PassoPipelinePreprocessador, ResultadoBuscaT
            ],
            avaliador_modelo: Avaliador,
            nome: str,
    ):
        self.__features_numericas = [
            "metragems",
            "quartos",
            "banheiros",
            "garagens",
            "banheiros_por_quarto",
        ]

        self.__features_categoricas = ["bairro"]
        self.__COLS_TO_LIST: Final[List[str]] = [
            "quartos",
            "banheiros",
            "garagens",
        ]
        self.__COLUNAS_TO_DROP: Final[List[str]] = [
            "precos",
            "id_casa",
            "url",
            "enderecos_apartamentos",
            "tipo_imovel",
        ]

        self.__preprocessamento_modelo = preprocessamento_modelo
        self.__treinamento_modelo = treinamento_modelo
        self.__avaliador_modelo = avaliador_modelo
        self.__nome = nome

    def realizar_preprocessamento_sklearn(
            self,
    ) -> ResultadoPreprocessamento[PassoPipelinePreprocessador]:

        self.__preprocessamento_modelo.caminho_arquivo = "processed.csv"
        self.__preprocessamento_modelo.cols_to_fix = self.__COLS_TO_LIST
        self.__preprocessamento_modelo.features_categoricas = (
            self.__features_categoricas
        )
        self.__preprocessamento_modelo.colunas_drop = self.__COLUNAS_TO_DROP
        self.__preprocessamento_modelo.features_numericas = (
            self.__features_numericas
        )

        x_train, x_test, y_train, y_test = (
            self.__preprocessamento_modelo.rodar_preprocessamento()
        )

        pipeline = self.__preprocessamento_modelo.preparar_modelo()

        return ResultadoPreprocessamento(
            x_train=x_train,
            x_test=x_test,
            y_train=y_train,
            y_test=y_test,
            pipeline=pipeline,
        )

    def rodar_modelo_sklearn(
            self,
            x_treino: DataFrame,
            x_teste: DataFrame,
            y_treino: Series,
            y_teste: Series,
            pipeline: PassoPipelinePreprocessador,
    ):
        self.__treinamento_modelo.passos_pipeline = pipeline
        self.__treinamento_modelo.treinar_modelo(x=x_treino, y=y_treino)

        pipeline_completo = self.__treinamento_modelo.dados_treinamento
        previsoes = self.__treinamento_modelo.predizer_modelo(x_test=x_teste)

        resultado_modelo = self.__avaliador_modelo.obter_resultados_modelo(
            pipeline=pipeline_completo,
            y_test=y_teste,
            y_pred=previsoes,
        )

        resultado_modelo["nome_modelo"] = self.__nome

        salvar_json(
            dados=resultado_modelo,
            diretorio="dados",
            nome_arquivo=f"avaliador_modelo_{self.__nome}_{datetime.now().strftime('%Y_%m_%d__%H_%M_%S')}",
        )

        valores_curva_validacao = (
            self.__avaliador_modelo.obter_dados_curva_validacao(
                pipeline=pipeline_completo,
                X_train=x_treino,
                y_train=y_treino,
            )
        )

        valores_curva_validacao["nome_modelo"] = self.__nome

        salvar_json(
            dados=valores_curva_validacao,
            diretorio="dados",
            nome_arquivo=f"valores_curva_validacao_{self.__nome}_{datetime.now().strftime('%Y_%m_%d__%H_%M_%S')}",
        )

        self.__avaliador_modelo.gerar_grafico_underfit_overfit(
            **valores_curva_validacao
        )

    def rodar_modelo_machine_learning(self, opcao: int):

        resultado = self.realizar_preprocessamento_sklearn()

        match opcao:
            case 1:
                self.rodar_modelo_sklearn(
                    x_treino=resultado.x_train,
                    x_teste=resultado.x_test,
                    y_treino=resultado.y_train,
                    y_teste=resultado.y_test,
                    pipeline=resultado.pipeline,
                )

            case 2:
                self.__treinamento_modelo.passos_pipeline = resultado.pipeline

                resultado_grid: ResultadoBuscaT = (
                    self.__treinamento_modelo.realizar_grid_search(
                        x=resultado.x_train,
                        y=resultado.y_train,
                    )
                )

                melhores_parametros = resultado_grid.best_params_
                melhores_parametros["nome_modelo"] = self.__nome

                salvar_json(
                    dados=melhores_parametros,
                    diretorio="dados",
                    nome_arquivo=f"realizar_tuning_parametros_{self.__nome}_{datetime.now().strftime('%Y_%m_%d__%H_%M_%S')}",
                )



modelos_machine_learning = [
    (
        "floresta_aleatoria",
        AvaliadorFlorestaAleatoria(),
        EstrategiaRegressaoRandomFlorest(),
    ),
    (
        "rede_neural",
        AvaliadorRedeNeural(),
        EstrategiaRegressaoRedeNeural(),
    ),
]

for nome, avaliador, modelo_machine_learning in modelos_machine_learning:
    pipeline = PipelineCompletaMachineLearning[
        PassoPipelineSklearn, GridSearchCV
    ](
        preprocessamento_modelo=PreprocessadorSklearn(),
        treinamento_modelo=modelo_machine_learning,
        avaliador_modelo=avaliador,
        nome=nome,
    )

    pipeline.rodar_modelo_machine_learning(opcao=1)