from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import FunctionTransformer, OneHotEncoder, StandardScaler

from src_machine_learning.processador.processador import Processador, ModeloMachineLearning
from sklearn.pipeline import Pipeline


class PrepocessadorSklearnn(Processador):



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

        feature_engineering = ('feature_engineering', FunctionTransformer(self.realizar_engenharia_atributos, validate=False))
        preprocessador = ('preprocessor', preprocessor)
        return [feature_engineering, preprocessador]

    def executar(self) -> ModeloMachineLearning:
        self.abrir_dataframe()
        self.fazer_processamento()
        dataframe = self.realizar_engenharia_atributos()
        return self.preparar_modelo()