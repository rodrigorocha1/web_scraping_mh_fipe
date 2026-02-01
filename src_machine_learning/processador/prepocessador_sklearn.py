import logging
import os
import re
from datetime import datetime

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import FunctionTransformer, OneHotEncoder, StandardScaler

from src_machine_learning.avaliador.avaliador import Avaliador
from src_machine_learning.config.variaveis import PassoPipelineSklearn
from src_machine_learning.estrategia_modelo.estrategia_modelo import EstrategiaModelo
from src_machine_learning.processador.processador import Processador
from src_machine_learning.utils.config_log import configurar_logging
from src_machine_learning.utils.utils import salvar_json

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

                logger.info(f'Treinando modelo {resultado}')
                dataframe = self.abrir_dataframe()
                dataframe = self.fazer_processamento(dataframe)

                x_train, x_test, y_train, y_test = self._separar_treino_teste(dataframe=dataframe)

                passos_pipeline = self._preparar_modelo()
                self._estrategia_modelo.pipeline = passos_pipeline

                self._estrategia_modelo.treinar_modelo(x=x_train, y=y_train)

                pipeline = self._estrategia_modelo.dados_treinamento

                previsoes = self._estrategia_modelo.predizer_modelo(x_test=x_test)

                flag_polinomial = self._estrategia_modelo.polinomial
                if flag_polinomial:
                    resultado = f'{resultado}_polinomial'

                dados = self._avaliador.obter_dados_curva_validacao(
                    pipeline=pipeline,
                    y_train=y_train,
                    X_train=x_train
                )
                os.makedirs(name=f'fig/gerar_grafico_over_under/{resultado}/', exist_ok=True)
                if dados is not None:
                    dados['data_coleta'] = datetime.now().strftime("%d/%m/%Y %H:%M:%S")
                    os.makedirs(name=f'dados/avaliador_modelo/{resultado}/', exist_ok=True)
                    salvar_json(
                        dados=dados,
                        diretorio=f'dados/avaliador_modelo/{resultado}',
                        nome_arquivo=f'avaliador_modelo_{resultado}',
                        identacao=4

                    )
                    dados['nome_modelo'] = resultado

                    self._avaliador.gerar_grafico_underfit_overfit(dados)

                resultado_previsoes_modelo_simples = self._avaliador.obter_resultados_modelo(
                    pipeline=pipeline,
                    y_test=y_test,
                    y_pred=previsoes
                )


                resultado_previsoes_modelo_simples['data_coleta'] = datetime.now().strftime("%d/%m/%Y %H:%M:%S")
                os.makedirs(name=f'dados/resultado_previsoes_modelo_simples/{resultado}/', exist_ok=True)
                salvar_json(
                    dados=resultado_previsoes_modelo_simples,
                    diretorio=f'dados/resultado_previsoes_modelo_simples/{resultado}',
                    nome_arquivo=f'resultado_previsoes_modelo_simples_{resultado}',
                    identacao=4

                )

            case 2:

                texto = self._estrategia_modelo.__class__.__name__
                parte = re.sub(r'^Estrategia', '', texto)
                nome_modelo = re.sub(r'(?<!^)([A-Z])', r'_\1', parte).lower()

                logger.info(f'Fazendo turing de parâmetos para {nome_modelo}')
                dataframe = self.abrir_dataframe()
                dataframe = self.fazer_processamento(dataframe)

                x_train, x_test, y_train, y_test = self._separar_treino_teste(dataframe=dataframe)

                x_completo = pd.concat([x_train, x_test], axis=0)
                y_completo = pd.concat([y_train, y_test], axis=0)
                # x_completo = self._realizar_engenharia_atributos_df(x_completo)
                # print(x_completo.describe())

                passos_pipeline = self._preparar_modelo()
                self._estrategia_modelo.pipeline = passos_pipeline
                resultado_grid = self._estrategia_modelo.realizar_grid_search(x=x_completo, y=y_completo)

                resultado_grid = resultado_grid.best_params_
                resultado_grid['data_coleta'] = datetime.now().strftime("%d/%m/%Y %H:%M:%S")


                os.makedirs(name=f'dados/resultado_turing_parametros/{nome_modelo}/', exist_ok=True)
                salvar_json(
                    dados=resultado_grid,
                    diretorio=f'dados/resultado_turing_parametros/{nome_modelo}',
                    nome_arquivo=f'resultado_turing_parametros_{nome_modelo}',
                    identacao=4

                )
            case 3:

                texto = self._estrategia_modelo.__class__.__name__
                parte = re.sub(r'^Estrategia', '', texto)
                nome_modelo = re.sub(r'(?<!^)([A-Z])', r'_\1', parte).lower()

                dataframe = self.abrir_dataframe()
                dataframe = self.fazer_processamento(dataframe)
                x_train, x_test, y_train, y_test = self._separar_treino_teste(dataframe=dataframe)
                x_trains = x_train.to_numpy()
                x_tests = x_test.to_numpy()
                y_trains = y_train.to_numpy()
                y_tests = y_test.to_numpy()

                x_completo = np.concatenate((x_trains, x_tests), axis=0)
                y_completo = np.concatenate((y_trains, y_tests), axis=0)
                passos_pipeline = self._preparar_modelo()
                self._estrategia_modelo.pipeline = passos_pipeline

                for i in range(30):
                    logging.info(f'Fazendo validação cruzada para {nome_modelo} - Iteração {i}')

                    resultado_validacao_cruzada = self._estrategia_modelo.realizar_validacao_cruzada(
                        x=x_completo,
                        y=y_completo,
                        iteracao=i
                    )
                    resultado_validacao_cruzada['data_coleta'] = datetime.now().strftime("%d/%m/%Y %H:%M:%S")
                    resultado_validacao_cruzada['nome_modelo']  =nome_modelo
                    os.makedirs(name=f'dados/resultados_validacao_cruzada/{nome_modelo}/', exist_ok=True)
                    salvar_json(
                        dados=resultado_validacao_cruzada,
                        diretorio=f'dados/resultados_validacao_cruzada/{nome_modelo}',
                        nome_arquivo=f'resultado_validacao_cruzada_{nome_modelo}'
                    )






