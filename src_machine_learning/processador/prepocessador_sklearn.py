from typing import TypeVar

from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import FunctionTransformer, OneHotEncoder, StandardScaler

from src_machine_learning.processador.processador import Processador, ModeloMachineLearning
from sklearn.pipeline import Pipeline

EstrategiaModelo = TypeVar('EstrategiaModelo')
class PrepocessadorSklearnn(Processador[EstrategiaModelo]):



    def __init__(self):
        super().__init__()

    def preparar_modelo(self, **kwargs) -> Pipeline:
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

        feature_engineering = ('feature_engineering', FunctionTransformer(self._realizar_engenharia_atributos_df, validate=False))
        preprocessador = ('preprocessor', preprocessor)
        return [feature_engineering, preprocessador]

    def executar(self) -> ModeloMachineLearning:
        dataframe = self.abrir_dataframe()
        print(dataframe)
        dataframe = self.fazer_processamento(dataframe=dataframe)
        dataframe = self._realizar_engenharia_atributos_df(dataframe=dataframe)
        print(dataframe)
        return self.preparar_modelo()