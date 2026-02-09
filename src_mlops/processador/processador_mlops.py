import logging
import re
from abc import ABC, abstractmethod
from typing import TypeVar, Generic

import pandas as pd
from sklearn.model_selection import train_test_split

from src_mlops.avaliador_mlops.avaliador import Avaliador
from src_mlops.config.variaveis import SeparacaoTreinoTeste
from src_mlops.estrategia_modelo.estrategia_modelo import EstrategiaModelo

pd.set_option("display.max_rows", 200)  # linhas máximas
pd.set_option("display.max_columns", 1000)  # colunas máximas
pd.set_option("display.width", 1000)  # largura do console
pd.set_option("display.max_colwidth", 40)  # largura do conteúdo
pd.set_option("display.float_format", "{:.2f}".format)

ModeloMachineLearning = TypeVar('ModeloMachineLearning')


class Processador(ABC, Generic[ModeloMachineLearning]):

    def __init__(self, estratregia_modelo: EstrategiaModelo, avaliador: Avaliador):
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
        self._colunas_categoricas = ['marca', 'tipo_combustivel', 'tipo_transmissao', 'turbo']
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
        self._features_numericas = [ 'motor_cilindrada', 'ano_modelo']
        self._features_categoricas = ['marca', 'modelo', 'tipo_combustivel','tipo_transmissao', 'turbo']
        self._estrategia_modelo = estratregia_modelo
        self._avaliador = avaliador

    @staticmethod
    def _extrair_transmissao(val):
        val = str(val).lower()
        if any(x in val for x in
               ['aut.', 'automático', 'automatico', 's-tronic', 'tip.', 'tiptronic', 'dsg', 'cvt', 'powershift']):
            return 'Automático'
        else:
            return 'Manual'

    @staticmethod
    def _extrair_turbo(val):
        val = str(val).lower()


        turbo_keywords = [
            'turbo', 'tfsi', 'tsi', 't-jet',  # Termos Originais
            'tb', 'biturbo', 'bi-turbo',  # Variações comuns
            'kompressor', 'compressor',  # Sobrealimentação
            'thp', 'tdi', 'cdi', 'cgi',  # Siglas de motores (Peugeot, VW/Audi, Mercedes)
            'ecoboost', 't-gdi', 'tgdi',  # Ford, Hyundai/Kia
            'jtd', 'hdi', 'd-4d', 'multijet',  # Diesel (Fiat, PSA, Toyota)
            'bluetec', 'tce', 'di-d', 'crdi',  # Outras tecnologias
            'duratorq', 'powerstroke',  # Pickups
            't270', 't200', 'td350', 'td380'  # Siglas de Torque (Jeep/Fiat)
        ]


        if any(k in val for k in turbo_keywords):
            return 'Sim'

        # Verificação 2: Regex para padrões de cilindrada + T (ex: "2.0T", "1.8 T")
        # \b = fronteira de palavra, \d = dígito, \s? = espaço opcional
        if re.search(r'\b[0-9]\.[0-9]\s?t\b', val):
            return 'Sim'

        return 'Não'

    @staticmethod
    def _extrair_motor(model_str: str):

        if pd.isna(model_str) or not isinstance(model_str, str):
            return None

        match = re.search(r'(\d\.\d)', model_str)
        if match:
            return float(match.group())
        return None

    @staticmethod
    def _extrair_potencia(model_str: str):
        match = re.search(r'(\d+)cv', str(model_str))
        if match:
            return int(match.group(1))
        return None

    def abrir_dataframe(self):
        dataframe = pd.read_csv(self.__caminho_arquivo)
        return dataframe

    def fazer_processamento(self, dataframe: pd.DataFrame):
        dataframe = dataframe[dataframe['Brand_Code'].isin(self.__marcas_ids)].copy()
        dataframe.rename(columns=self._colunas_rename, inplace=True)

        dataframe['preco'] = dataframe['preco'].astype(str).str.replace('R$ ', '', regex=False)
        dataframe['preco'] = dataframe['preco'].str.replace('.', '', regex=False)
        dataframe['preco'] = dataframe['preco'].str.replace(',', '.', regex=False)
        dataframe['preco'] = pd.to_numeric(dataframe['preco'])
        dataframe = dataframe[dataframe['preco'] <= 500000.00]
        dataframe['ano_modelo'] = dataframe['codigo_ano'].str.split('-').str[0].astype(int)
        dataframe = dataframe[dataframe['ano_modelo'] >= 2000]

        dataframe.drop(
            columns=['tipo', 'codigo_marca', 'codigo_modelo', 'codigo_ano',
                     'ano_combustivel', 'codigo_fipe', 'sigla_combustivel', 'Month'],
            inplace=True
        )

        return dataframe

    def _realizar_engenharia_atributos_df(self, dataframe: pd.DataFrame) -> pd.DataFrame:

        dataframe['motor_cilindrada'] = dataframe['modelo'].apply(self._extrair_motor)
        dataframe["ano_modelo"] = dataframe["ano_modelo"].replace(32000, 2026)
        dataframe['tipo_transmissao'] = dataframe['modelo'].apply(self._extrair_transmissao)
        dataframe['turbo'] = dataframe['modelo'].apply(self._extrair_turbo)
        dataframe[self._colunas_categoricas] = dataframe[self._colunas_categoricas].astype('category')

        return dataframe

    @staticmethod
    def _separar_treino_teste(dataframe: pd.DataFrame) -> SeparacaoTreinoTeste:
        x = dataframe.drop(columns=['preco'])
        y = dataframe['preco']
        x_train, x_test, y_train, y_test = train_test_split(
            x,
            y,
            test_size=0.3,
            random_state=42
        )
        return x_train, x_test, y_train, y_test

    @abstractmethod
    def executar(self, opcao: int) -> ModeloMachineLearning:
        pass

    @abstractmethod
    def _preparar_modelo(self, **kwargs) -> ModeloMachineLearning:
        pass
