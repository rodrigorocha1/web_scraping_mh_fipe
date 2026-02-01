from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import FunctionTransformer, OneHotEncoder, StandardScaler

from src_machine_learning.avaliador.avaliador import Avaliador
from src_machine_learning.config.variaveis import PassoPipelineSklearn
from src_machine_learning.estrategia_modelo.estrategia_modelo import EstrategiaModelo
from src_machine_learning.processador.processador import Processador


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
                dataframe = self.abrir_dataframe()
                dataframe = self.fazer_processamento(dataframe)

                x_train, x_test, y_train, y_test = self._separar_treino_teste(dataframe=dataframe)

                passos_pipeline = self._preparar_modelo()
                self._estrategia_modelo.pipeline = passos_pipeline

                self._estrategia_modelo.treinar_modelo(x=x_train, y=y_train)

                pipeline = self._estrategia_modelo.dados_treinamento


                previsoes = self._estrategia_modelo.predizer_modelo(x_test=x_test)

                dados = self._avaliador.obter_resultados_modelo(
                    pipeline=pipeline,
                    y_test=y_test,
                    y_pred=previsoes
                )
                print(self._estrategia_modelo.__dict__)
