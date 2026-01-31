import re

import pandas as pd

pd.set_option("display.max_rows", 200)  # linhas máximas
pd.set_option("display.max_columns", 1000)  # colunas máximas
pd.set_option("display.width", 1000)  # largura do console
pd.set_option("display.max_colwidth", 40)  # largura do conteúdo
pd.set_option("display.float_format", "{:.2f}".format)  # floats formatados
import pandas as pd

# Lista de IDs das marcas escolhidas conforme sua solicitação
marcas_ids = [
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

# Carregar o arquivo original
# Certifique-se de que o arquivo 'tabela-fipe-329.csv' esteja na mesma pasta
df = pd.read_csv('tabela-fipe-329.csv')

# 1. Filtrar pelas marcas selecionadas
df_ml = df[df['Brand_Code'].isin(marcas_ids)].copy()

df_ml.rename(columns={
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

}, inplace=True)

df_ml['ano_modelo'] = df_ml['codigo_ano'].str.split('-').str[0]
df_ml['ano_modelo'] = df_ml['ano_modelo'].astype(int)

df_ml = df_ml[(df_ml['ano_modelo'] >= 2000)]


# 4. Extract Engine Size (e.g., 1.0, 2.0, 3.2)
def extrair_motor(model_str):
    # Verifica se o valor é nulo ou não é uma string
    if pd.isna(model_str) or not isinstance(model_str, str):
        return None

    match = re.search(r'(\d\.\d)', model_str)
    if match:
        return float(match.group())
    return None


def extrair_potencia(model_str):
    match = re.search(r'(\d+)cv', str(model_str))
    if match:
        return int(match.group(1))
    return None


df_ml['motor_cilindrada'] = df_ml['modelo'].apply(extrair_motor)

print(df_ml['motor_cilindrada'].isna().sum())

df_ml['preco'] = df_ml['preco'].astype(str).str.replace('R$ ', '', regex=False)
df_ml['preco'] = df_ml['preco'].str.replace('.', '', regex=False)
df_ml['preco'] = df_ml['preco'].str.replace(',', '.', regex=False)
df_ml['preco'] = pd.to_numeric(df_ml['preco'])
df_ml = df_ml[df_ml['preco'] <= 500000.00]
colunas_categoricas = ['codigo_marca', 'marca', 'sigla_combustivel', 'tipo_combustivel']
df_ml[colunas_categoricas] = df_ml[colunas_categoricas].astype('category')

df_ml.drop(columns=['tipo', 'codigo_marca', 'codigo_modelo', 'ano_combustivel', 'codigo_fipe', 'sigla_combustivel', 'Month'],  inplace=True)

print(df_ml.duplicated().sum())
print(df_ml.describe())
print(df_ml.head())
print(df_ml.dtypes)


