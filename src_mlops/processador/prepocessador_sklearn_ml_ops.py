import logging
import pickle
import re
from typing import List

import mlflow
import pandas as pd
from mlflow.models.signature import infer_signature
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import FunctionTransformer, OneHotEncoder, StandardScaler

from src_mlops.avaliador_mlops.avaliador import Avaliador
from src_mlops.config.variaveis import PassoPipelineSklearn
from src_mlops.estrategia_modelo.estrategia_modelo import EstrategiaModelo
from src_mlops.processador.processador_mlops import Processador
from src_mlops.utils.config_log import configurar_logging
from src_mlops.utils.mlflow_config import configurar_mlflow

configurar_logging()

logger = logging.getLogger(__name__)


class PrepocessadorSklearnn(Processador):

    def __init__(self, estratregia_modelo: EstrategiaModelo, avaliador: Avaliador,
                 modelos_votacao: List[EstrategiaModelo] = None):
        super().__init__(estratregia_modelo=estratregia_modelo, avaliador=avaliador)
        self._x_train = None
        self._y_train = None
        self.__modelos_votacao = modelos_votacao

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
                EXPERIMENT_NAME = f"treinamento_simples_{resultado}_metricas"
                configurar_mlflow(experiment_name=EXPERIMENT_NAME, tracking_uri=MLFLOW_URI)

                logger.info(f'Treinando modelo {resultado}')

                # Carrega e processa dataframe
                dataframe = self.abrir_dataframe()
                dataframe = self.fazer_processamento(dataframe)

                x_train, x_test, y_train, y_test = self._separar_treino_teste(dataframe=dataframe)
                self._x_train = x_train
                self._y_train = y_train

                # Prepara pipeline
                passos_pipeline = self._preparar_modelo()
                self._estrategia_modelo.pipeline = passos_pipeline


                artifact_name = f"modelo_pipeline_{resultado}"
                mlflow.sklearn.autolog()  # ativa autolog

                with mlflow.start_run(run_name=f'treinamento_simples_{resultado}') as run:
                    # Treina modelo
                    self._estrategia_modelo.treinar_modelo(x=x_train, y=y_train)

                    # Recupera pipeline treinada
                    pipeline = self._estrategia_modelo.dados_treinamento

                    # Faz previsões para assinatura
                    previsoes = self._estrategia_modelo.predizer_modelo(x_test=x_test)
                    signature = infer_signature(x_train, previsoes)

                    model_info = mlflow.sklearn.log_model(
                        sk_model=pipeline,  # Se isso for um sklearn.pipeline.Pipeline, está correto
                        artifact_path="model",
                        signature=signature,
                        registered_model_name=artifact_name
                    )
                    # Log pipeline completo no MLflow

                    eval_data = x_test.copy()
                    eval_data["preco"] = y_test
                    for col in self._features_categoricas:
                        eval_data[col] = eval_data[col].astype(object)
                    for col in self._features_numericas:
                        eval_data[col] = eval_data[col].astype(float)
                    eval_data['ano_modelo'] = eval_data['ano_modelo'].astype("int64")
                    result = mlflow.models.evaluate(

                        model_info.model_uri,

                        eval_data,

                        targets="preco",

                        model_type="regressor"

                    )

                    print(f"MAE: {result.metrics['mean_absolute_error']:.3f}")
                    print(f"RMSE: {result.metrics['root_mean_squared_error']:.3f}")
                    print(f"R² Score: {result.metrics['r2_score']:.3f}")
                    metricas = self._avaliador.obter_dados_curva_validacao(
                        pipeline=pipeline,
                        X_train=self._x_train,
                        y_train=self._y_train
                    )
                    self._avaliador.gerar_grafico_underfit_overfit(metricas)

                    registered_model_name = artifact_name

                    logger.info(f"Modelo registrado como: {registered_model_name}")

            case 2:

                texto = self._estrategia_modelo.__class__.__name__
                parte = re.sub(r'^Estrategia', '', texto)
                nome_modelo = re.sub(r'(?<!^)([A-Z])', r'_\1', parte).lower()
                MLFLOW_URI = "http://172.25.0.5:5000"

                EXPERIMENT_NAME = f"turing_parametros_{nome_modelo}_v2"
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

                    print(f"Best params: {resultado_grid.best_params_}")
                    print(f"Best CV score: {resultado_grid.best_score_:.3f}")
                    try:
                        print(f"Test score: {resultado_grid:.3f}")
                    except:
                        pass

                    mlflow.sklearn.log_model(
                        sk_model=resultado_grid.best_params_,
                        artifact_path="model",
                        registered_model_name=f"{nome_modelo}_v2",
                        pip_requirements=[
                            "scikit-learn==1.4.2",
                            "pandas",
                            "numpy"
                        ]
                    )

            case 3:

                texto = self._estrategia_modelo.__class__.__name__
                parte = re.sub(r'^Estrategia', '', texto)
                nome_modelo = re.sub(r'(?<!^)([A-Z])', r'_\1', parte).lower()

                MLFLOW_URI = "http://172.25.0.5:5000"

                EXPERIMENT_NAME = f"validacao_cruzada_{nome_modelo}_v2"
                configurar_mlflow(experiment_name=EXPERIMENT_NAME, tracking_uri=MLFLOW_URI)

                dataframe = self.abrir_dataframe()
                dataframe = self.fazer_processamento(dataframe)
                x_train, x_test, y_train, y_test = self._separar_treino_teste(dataframe=dataframe)
                # x_trains = x_train.to_numpy()
                # x_tests = x_test.to_numpy()
                # y_trains = y_train.to_numpy()
                # y_tests = y_test.to_numpy()

                dados_para_salvar = {
                    'x_train': x_train,
                    'x_test': x_test,
                    'y_train': y_train,
                    'y_test': y_test

                }

                with open('dados_completos.pkl', 'wb') as arquivo:
                    pickle.dump(dados_para_salvar, arquivo)

                print("Dados gravados com sucesso em 'dados_completos.pkl'!")

                x_completo = pd.concat((x_train, x_test), axis=0)
                y_completo = pd.concat((y_train, y_test), axis=0)

                passos_pipeline = self._preparar_modelo()
                self._estrategia_modelo.pipeline = passos_pipeline

                for i in range(30):
                    with mlflow.start_run(run_name=f"cv_iteracao_{i}", nested=True):

                        logging.info(
                            f'Fazendo validação cruzada para {nome_modelo} - Iteração {i}'
                        )

                        resultado_validacao_cruzada = self._estrategia_modelo.realizar_validacao_cruzada(
                            x=x_completo,
                            y=y_completo,
                            iteracao=i
                        )

                        # Log parâmetros do pipeline
                        for nome, valor in passos_pipeline:
                            if hasattr(valor, 'get_params'):
                                params = valor.get_params()
                                for p_nome, p_valor in params.items():
                                    mlflow.log_param(f"{nome}_{p_nome}", p_valor)

                        # Métricas médias
                        for metrica, valor in resultado_validacao_cruzada["mean_scores"].items():
                            mlflow.log_metric(metrica, valor)

                        # Métricas por fold
                        for idx, rmse in enumerate(resultado_validacao_cruzada["rmse_folds"]):
                            mlflow.log_metric(f"rmse_fold_{idx}", rmse)

                        mlflow.set_tag("iteracao", i)
                        mlflow.set_tag("nome_modelo", nome_modelo)
                        print(self._estrategia_modelo.pipeline)
                        # # 🔥 REGISTRAR MODELO
                        # mlflow.sklearn.log_model(
                        #     sk_model=self._estrategia_modelo.pipeline,
                        #     artifact_path="model",
                        #     registered_model_name=f"{nome_modelo}_cv_v2",
                        #     pip_requirements=[
                        #         "scikit-learn==1.4.2",
                        #         "pandas",
                        #         "numpy"
                        #     ]
                        # )
                        # break

            case 4:
                print('votação com os melhores modelos')
                texto = self._estrategia_modelo.__class__.__name__
                parte = re.sub(r'^Estrategia', '', texto)
                nome_modelo = re.sub(r'(?<!^)([A-Z])', r'_\1', parte).lower()

                MLFLOW_URI = "http://172.25.0.5:5000"

                EXPERIMENT_NAME = f"regressao_votacao_v2"
                configurar_mlflow(experiment_name=EXPERIMENT_NAME, tracking_uri=MLFLOW_URI)

                dataframe = self.abrir_dataframe()
                dataframe = self.fazer_processamento(dataframe)
                x_train, x_test, y_train, y_test = self._separar_treino_teste(dataframe=dataframe)
                passos_pipeline = self._preparar_modelo()
                x_completo = pd.concat([x_train, x_test], axis=0)
                y_completo = pd.concat([y_train, y_test], axis=0)

                dados_para_salvar = {
                    'x_train': x_train,
                    'x_test': x_test,
                    'y_train': y_train,
                    'y_test': y_test

                }

                with open('dados_completos.pkl', 'wb') as arquivo:
                    pickle.dump(dados_para_salvar, arquivo)

                print("Dados gravados com sucesso em 'dados_completos.pkl'!")
