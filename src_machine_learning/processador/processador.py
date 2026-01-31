import re
from abc import ABC, abstractmethod
from typing import TypeVar, Generic

import pandas as pd

pd.set_option("display.max_rows", 200)  # linhas máximas
pd.set_option("display.max_columns", 1000)  # colunas máximas
pd.set_option("display.width", 1000)  # largura do console
pd.set_option("display.max_colwidth", 40)  # largura do conteúdo
pd.set_option("display.float_format", "{:.2f}".format)

ModeloMachineLearning = TypeVar('ModeloMachineLearning')


class Processador(ABC, Generic[ModeloMachineLearning]):

    def __init__(self):
        self.__caminho_arquivo = "tabela-fipe-329.csv"
        self.__marcas_ids = [
            6,  # Audi
            238,  # BYD
            23,  # GM - Chevrolet
            13,  # Citroën
            21,  # Fiat
            22,  # Ford
            25,  # Honda
            26,  # Hyundai
            208,  # IVECO
            177,  # JAC
            29,  # Jeep
            31,  # Kia Motors
            39,  # Mercedes-Benz
            41,  # Mitsubishi
            43,  # Nissan
            44,  # Peugeot
            48,  # Renault
            56,  # Toyota
            59,  # VW - VolksWagen
            57,  # Troller
            58  # Volvo
        ]
        self._colunas_categoricas = ['codigo_marca', 'marca', 'sigla_combustivel', 'tipo_combustivel']
        self._colunas_rename = {
            'Type': 'tipo',
            'Brand_Code': 'codigo_marca',
            'Brand_Value': 'marca',
            'Model_Code': 'codigo_modelo',
            'Model_value': 'modelo',
            'Year_Code': 'codigo_ano',
            'Year_Value': 'ano_combustivel',
            'Fipe_Code': 'codigo_fipe',
            'Fuel_Letter': 'sigla_combustivel',
            'Fuel_Type': 'tipo_combustivel',
            'Price': 'preco',

        }
        self._features_numericas = ['preco', 'motor_cilindrada', 'ano_modelo']
        self._features_categoricas = ['marca', 'modelo', 'tipo_combustivel']
        self._dataframe = None

    @staticmethod
    def _extrair_motor(model_str: str):
        # Verifica se o valor é nulo ou não é uma string
        if pd.isna(model_str) or not isinstance(model_str, str):
            return None

        match = re.search(r'(\d\.\d)', model_str)
        if match:
            return float(match.group())
        return None

    @staticmethod
    def extrair_potencia(model_str: str):
        match = re.search(r'(\d+)cv', str(model_str))
        if match:
            return int(match.group(1))
        return None

    def abrir_dataframe(self):
        self._dataframe = pd.read_csv(self.__caminho_arquivo)

    def fazer_processamento(self):
        self._dataframe = self._dataframe[self._dataframe['Brand_Code'].isin(self.__marcas_ids)].copy()
        self._dataframe.rename(columns=self._colunas_rename, inplace=True)

        self._dataframe['preco'] = self._dataframe['preco'].astype(str).str.replace('R$ ', '', regex=False)
        self._dataframe['preco'] = self._dataframe['preco'].str.replace('.', '', regex=False)
        self._dataframe['preco'] = self._dataframe['preco'].str.replace(',', '.', regex=False)
        self._dataframe['preco'] = pd.to_numeric(self._dataframe['preco'])
        self._dataframe = self._dataframe[self._dataframe['preco'] <= 500000.00]
        self._dataframe[self._colunas_categoricas] = self._dataframe[self._colunas_categoricas].astype('category')

    def engenharia_atributos_df(self) -> pd.DataFrame:
        X = self._dataframe.copy()
        X['motor_cilindrada'] = X['modelo'].apply(self._extrair_motor)
        X['ano_modelo'] = X['codigo_ano'].str.split('-').str[0].astype(int)
        X = X[X['ano_modelo'] >= 2000]
        X.drop(
            columns=['tipo', 'codigo_marca', 'codigo_modelo', 'codigo_ano',
                     'ano_combustivel', 'codigo_fipe', 'sigla_combustivel', 'Month'],
            inplace=True
        )
        return X

    @abstractmethod
    def executar(self) -> ModeloMachineLearning:
        pass

    @abstractmethod
    def preparar_modelo(self, **kwargs) -> ModeloMachineLearning:
        pass
