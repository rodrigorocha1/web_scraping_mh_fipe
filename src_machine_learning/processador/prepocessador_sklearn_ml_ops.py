import logging
import re
from datetime import datetime

import mlflow
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import FunctionTransformer, OneHotEncoder, StandardScaler

from src_machine_learning.avaliador_mlops.avaliador import Avaliador
from src_machine_learning.config.variaveis import PassoPipelineSklearn
from src_machine_learning.estrategia_modelo.estrategia_modelo import EstrategiaModelo
from src_machine_learning.processador.processador_mlops import Processador
from src_machine_learning.utils.config_log import configurar_logging
from src_machine_learning.utils.mlflow_config import configurar_mlflow

configurar_logging()

logger = logging.getLogger(__name__)


class PrepocessadorSklearnn(Processador):

    def __init__(self, estratregia_modelo: EstrategiaModelo, avaliador: Avaliador):
        super().__init__(estratregia_modelo=estratregia_modelo, avaliador=avaliador)

    def _preparar_modelo(self, **kwargs) -> PassoPipelineSklearn:
        num_pipeline = Pipeline(steps=[
            ('imputer', SimpleImputer(strategy='median')),
            ('scaler', StandardScaler())
        ])

        preprocessor = ColumnTransformer(
            transformers=[
                ('num', num_pipeline, self._features_numericas),
                ('cat', OneHotEncoder(
                    handle_unknown='ignore',
                    min_frequency=30,
                    sparse_output=False
                ), self._features_categoricas)
            ],
            verbose_feature_names_out=False
        )

        feature_engineering = ('feature_engineering',
                               FunctionTransformer(self._realizar_engenharia_atributos_df, validate=False))
        preprocessador = ('preprocessor', preprocessor)
        return [feature_engineering, preprocessador]

    def executar(self, opcao: int):

        match opcao:
            case 1:
                texto = self._estrategia_modelo.__class__.__name__
                parte = re.sub(r'^Estrategia', '', texto)
                resultado = re.sub(r'(?<!^)([A-Z])', r'_\1', parte).lower()
                MLFLOW_URI = "http://172.25.0.5:5000"

                EXPERIMENT_NAME = f"treinamento_simples_{resultado}_v2"
                configurar_mlflow(experiment_name=EXPERIMENT_NAME, tracking_uri=MLFLOW_URI)

                logger.info(f'Treinando modelo {resultado}')
                dataframe = self.abrir_dataframe()
                dataframe = self.fazer_processamento(dataframe)


                x_train, x_test, y_train, y_test = self._separar_treino_teste(dataframe=dataframe)

                passos_pipeline = self._preparar_modelo()

                self._estrategia_modelo.pipeline = passos_pipeline
                artifact_name = f"modelo_pipeline_{resultado}"
                mlflow.sklearn.autolog()

                with mlflow.start_run(run_name=f'treinamento_simples_{resultado}') as run:
                    self._estrategia_modelo.treinar_modelo(x=x_train, y=y_train)

                    pipeline = self._estrategia_modelo.dados_treinamento
                    regressor = pipeline.named_steps['regressor']

                    previsoes = self._estrategia_modelo.predizer_modelo(x_test=x_test)
                    # Loga o pipeline completo
                    print(self._estrategia_modelo.pipeline)

                    model_uri = f"runs:/{run.info.run_id}/{artifact_name}"
                    registered_model_name = artifact_name

                    mlflow.sklearn.log_model(
                        sk_model=self._estrategia_modelo.pipeline,
                        artifact_path=f"modelo_pipeline_{resultado}"
                    )
                    mlflow.register_model(model_uri, registered_model_name)

                    flag_polinomial = self._estrategia_modelo.polinomial
                    # if flag_polinomial:
                    #     resultado = f'{resultado}_polinomial'
                    #
                    #
                    #
                    dados = self._avaliador.obter_dados_curva_validacao(
                        pipeline=pipeline,
                        y_train=y_train,
                        X_train=x_train
                    )

                    if dados is not None:
                        media_se_lista = lambda x: sum(x) / len(x) if isinstance(x, list) and len(x) > 0 else x

                        for metric_name, value in dados.items():
                            print(metric_name, value)
                            mlflow.log_metric(metric_name, media_se_lista(value))
                        dados['nome_modelo'] = resultado
                        dados['data_coleta'] = datetime.now().strftime("%d/%m/%Y %H:%M:%S")

                        self._avaliador.gerar_grafico_underfit_overfit(dados)

                        self._avaliador.gerar_grafico_underfit_overfit(dados)

                    resultado_previsoes_modelo_simples = self._avaliador.obter_resultados_modelo(
                        pipeline=pipeline,
                        y_test=y_test,
                        y_pred=previsoes
                    )
                    print(resultado_previsoes_modelo_simples)

                    for dado in resultado_previsoes_modelo_simples.items():
                        if isinstance(dado[1], dict):
                            print('verdadeiro')
                            for item in dado[1].items():
                                mlflow.log_metric(item[0], item[1])
                        else:
                            print('falso')
                            mlflow.log_metric(dado[0], dado[1])

            case 2:

                texto = self._estrategia_modelo.__class__.__name__
                parte = re.sub(r'^Estrategia', '', texto)
                nome_modelo = re.sub(r'(?<!^)([A-Z])', r'_\1', parte).lower()
                MLFLOW_URI = "http://172.25.0.5:5000"

                EXPERIMENT_NAME = f"treinamento_simples_{nome_modelo}_v2"
                configurar_mlflow(experiment_name=EXPERIMENT_NAME, tracking_uri=MLFLOW_URI)

                logger.info(f'Fazendo turing de parâmetos para {nome_modelo}')
                dataframe = self.abrir_dataframe()
                dataframe = self.fazer_processamento(dataframe)

                x_train, x_test, y_train, y_test = self._separar_treino_teste(dataframe=dataframe)

                x_completo = pd.concat([x_train, x_test], axis=0)
                y_completo = pd.concat([y_train, y_test], axis=0)
                # x_completo = self._realizar_engenharia_atributos_df(x_completo)
                # print(x_completo.describe())
                flag_polinomial = self._estrategia_modelo.polinomial
                if flag_polinomial:
                    nome_modelo = f'{nome_modelo}_polinomial'

                mlflow.sklearn.autolog()
                with mlflow.start_run(run_name=f'Turing de Hiperparâmetros de {nome_modelo}'):

                    passos_pipeline = self._preparar_modelo()
                    self._estrategia_modelo.pipeline = passos_pipeline
                    resultado_grid = self._estrategia_modelo.realizar_grid_search(x=x_completo, y=y_completo)

                    resultado_grid = resultado_grid.best_params_


            # case 3:
            #
            #     texto = self._estrategia_modelo.__class__.__name__
            #     parte = re.sub(r'^Estrategia', '', texto)
            #     nome_modelo = re.sub(r'(?<!^)([A-Z])', r'_\1', parte).lower()
            #
            #     dataframe = self.abrir_dataframe()
            #     dataframe = self.fazer_processamento(dataframe)
            #     x_train, x_test, y_train, y_test = self._separar_treino_teste(dataframe=dataframe)
            #     # x_trains = x_train.to_numpy()
            #     # x_tests = x_test.to_numpy()
            #     # y_trains = y_train.to_numpy()
            #     # y_tests = y_test.to_numpy()
            #
            #     x_completo = pd.concat((x_train, x_test), axis=0)
            #     y_completo = pd.concat((y_train, y_test), axis=0)
            #     print(x_completo)
            #     passos_pipeline = self._preparar_modelo()
            #     self._estrategia_modelo.pipeline = passos_pipeline
            #
            #     for i in range(30):
            #         logging.info(f'Fazendo validação cruzada para {nome_modelo} - Iteração {i}')
            #
            #         resultado_validacao_cruzada = self._estrategia_modelo.realizar_validacao_cruzada(
            #             x=x_completo,
            #             y=y_completo,
            #             iteracao=i
            #         )

                    # for nome, valor in passos_pipeline:
                    #     if hasattr(valor, 'get_params'):
                    #         params = valor.get_params()
                    #         for p_nome, p_valor in params.items():
                    #             mlflow.log_param(f"{nome}_{p_nome}", p_valor)
                    #
                    #     # Métricas médias
                    # for metrica, valor in resultado_validacao_cruzada["mean_scores"].items():
                    #     mlflow.log_metric(metrica, valor)
                    #
                    #     # Métricas por fold (RMSE)
                    # for idx, rmse in enumerate(resultado_validacao_cruzada["rmse_folds"]):
                    #     mlflow.log_metric(f"rmse_fold_{idx}", rmse)
                    #
                    # mlflow.set_tag("iteracao", i)
                    # mlflow.set_tag("nome_modelo", nome_modelo)

            #         flag_polinomial = self._estrategia_modelo.polinomial
            #         if flag_polinomial:
            #             nome_modelo = f'{nome_modelo}_polinomial'
