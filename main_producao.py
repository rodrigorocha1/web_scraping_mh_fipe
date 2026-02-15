import math
import pickle
import re
import ast

import mlflow
import numpy as np
import pandas as pd
from mlflow.models import infer_signature
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestRegressor, VotingRegressor
from sklearn.impute import SimpleImputer
from sklearn.metrics import mean_squared_error
from sklearn.neural_network import MLPRegressor
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, OneHotEncoder, FunctionTransformer
from sklearn.tree import DecisionTreeRegressor

from src_machine_learning.utils.utils import carregar_dados_yaml_lista
from src_mlops.utils.mlflow_config import configurar_mlflow

# ------------------------ Config pandas ------------------------
pd.set_option("display.max_rows", 200)
pd.set_option("display.max_columns", 1000)
pd.set_option("display.width", 1000)
pd.set_option("display.max_colwidth", 40)
pd.set_option("display.float_format", "{:.2f}".format)

# ------------------------ Configs ------------------------
colunas_categoricas = ['marca', 'tipo_combustivel', 'tipo_transmissao', 'turbo']
colunas_rename = {
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
features_numericas = ['motor_cilindrada', 'ano_modelo']
features_categoricas = ['marca', 'modelo', 'tipo_combustivel', 'tipo_transmissao', 'turbo']

# ------------------------ Carregando dados ------------------------
with open('dados_completos.pkl', 'rb') as arquivo:
    dados_carregados = pickle.load(arquivo)

# Mantendo DataFrames e Series
x_train = pd.DataFrame(dados_carregados['x_train'])
x_test = pd.DataFrame(dados_carregados['x_test'])
y_train = pd.Series(dados_carregados['y_train'])
y_test = pd.Series(dados_carregados['y_test'])

print(x_train)
print(x_train)
# ------------------------ Funções auxiliares ------------------------
def extrair_motor(model_str: str):
    if pd.isna(model_str) or not isinstance(model_str, str):
        return None
    match = re.search(r'(\d\.\d)', model_str)
    if match:
        return float(match.group())
    return None


def extrair_transmissao(val):
    val = str(val).lower()
    if any(x in val for x in
           ['aut.', 'automático', 'automatico', 's-tronic', 'tip.', 'tiptronic', 'dsg', 'cvt', 'powershift']):
        return 'Automático'
    return 'Manual'


def extrair_turbo(val):
    val = str(val).lower()
    turbo_keywords = [
        'turbo', 'tfsi', 'tsi', 't-jet', 'tb', 'biturbo', 'bi-turbo',
        'kompressor', 'compressor', 'thp', 'tdi', 'cdi', 'cgi',
        'ecoboost', 't-gdi', 'tgdi', 'jtd', 'hdi', 'd-4d', 'multijet',
        'bluetec', 'tce', 'di-d', 'crdi', 'duratorq', 'powerstroke',
        't270', 't200', 'td350', 'td380'
    ]
    if any(k in val for k in turbo_keywords):
        return 'Sim'
    if re.search(r'\b[0-9]\.[0-9]\s?t\b', val):
        return 'Sim'
    return 'Não'


def realizar_engenharia_atributos_df(dataframe):
    """
    Aplica engenharia de atributos no DataFrame.
    Se receber um NumPy array, converte para DataFrame.
    """
    if isinstance(dataframe, np.ndarray):
        # Converter para DataFrame assumindo as colunas originais
        dataframe = pd.DataFrame(dataframe, columns=x_train.columns)

    dataframe['motor_cilindrada'] = dataframe['modelo'].apply(extrair_motor)
    dataframe["ano_modelo"] = dataframe["ano_modelo"].replace(32000, 2026)
    dataframe['tipo_transmissao'] = dataframe['modelo'].apply(extrair_transmissao)
    dataframe['turbo'] = dataframe['modelo'].apply(extrair_turbo)
    dataframe[colunas_categoricas] = dataframe[colunas_categoricas].astype('category')
    return dataframe


# ------------------------ Pipeline ------------------------
num_pipeline = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='median')),
    ('scaler', StandardScaler())
])

preprocessor = ColumnTransformer(
    transformers=[
        ('num', num_pipeline, features_numericas),
        ('cat', OneHotEncoder(handle_unknown='ignore', min_frequency=30, sparse_output=False), features_categoricas)
    ],
    verbose_feature_names_out=False
)

pipeline_completo = [
    ('feature_engineering', FunctionTransformer(realizar_engenharia_atributos_df, validate=False)),
    ('preprocessor', preprocessor)
]

# ------------------------ MLflow ------------------------
MLFLOW_URI = "http://localhost:5000"
EXPERIMENT_NAME = "modelo_pronto_votacao"
configurar_mlflow(experiment_name=EXPERIMENT_NAME, tracking_uri=MLFLOW_URI)


def converter_parametros_mlflow(parametros: dict, tipo_modelo: str):
    """Converte parâmetros vindos do MLflow para os tipos corretos do sklearn."""
    p = parametros.copy()
    for chave, valor in p.items():
        if isinstance(valor, str):
            try:
                if '.' in valor:
                    p[chave] = float(valor)
                else:
                    p[chave] = int(valor)
            except ValueError:
                if valor.lower() == 'true':
                    p[chave] = True
                elif valor.lower() == 'false':
                    p[chave] = False
                elif valor.lower() == 'none':
                    p[chave] = None
                else:
                    try:
                        val = ast.literal_eval(valor)
                        if chave == 'hidden_layer_sizes' and isinstance(val, list):
                            p[chave] = tuple(val)
                        else:
                            p[chave] = val
                    except:
                        p[chave] = valor
    return p


# ------------------------ Carrega parâmetros ------------------------
parametros_list = carregar_dados_yaml_lista(parametro_modelo='parametros_treinamento_simples')
parametros_random_florest = parametros_list[0]['parametros']
parametros_arvore_decisao = parametros_list[1]['parametros']
parametros_rede_neural = parametros_list[3]['parametros']

estimadores = []
def to_writeable_array(df_or_series):
    """
    Converte DataFrame ou Series para numpy array writeable
    """
    arr = np.array(df_or_series, copy=True)
    arr.setflags(write=1)  # garante que seja writeable
    return arr

# Use isso antes de fit/predict:
X_train = to_writeable_array(x_train)
X_test = to_writeable_array(x_test)
Y_train = to_writeable_array(y_train)
Y_test = to_writeable_array(y_test)

x_train = x_train.copy()
X_test = X_test.copy()
Y_train = Y_train.copy()
Y_test = Y_test.copy()
# ------------------------ Treinamento MLP ------------------------
with mlflow.start_run(run_name='regressao_rede_neural'):
    pipeline_rede_neural = Pipeline(steps=pipeline_completo + [
        ('modelo', MLPRegressor(**parametros_rede_neural))
    ])
    pipeline_rede_neural.fit(X=x_train.copy(), y=y_train.copy())
    previsoes = pipeline_rede_neural.predict(x_test.copy())
    mlflow.log_metric("rmse", math.sqrt(mean_squared_error(y_test.copy(), previsoes)))
    mlflow.sklearn.log_model(pipeline_rede_neural, "modelo_rede_neural")
    estimadores.append(("rrn", pipeline_rede_neural.named_steps['modelo']))
    mlflow.sklearn.log_model(
        sk_model=pipeline_rede_neural,
        artifact_path="model_votacao_rede_neural",
        registered_model_name="VotingRegressorPipelineRedeNeural"
    )

# ------------------------ Treinamento RandomForest ------------------------
with mlflow.start_run(run_name='regressao_random_florest'):
    pipeline_random_florest = Pipeline(steps=pipeline_completo + [
        ('modelo', RandomForestRegressor(**parametros_random_florest))
    ])
    pipeline_random_florest.fit(X=x_train.copy(), y=y_train.copy())
    previsoes = pipeline_random_florest.predict(x_test.copy())
    mlflow.log_metric("rmse", math.sqrt(mean_squared_error(y_test.copy(), previsoes)))
    mlflow.sklearn.log_model(pipeline_random_florest, "modelo_random_florest")
    estimadores.append(("rrf", pipeline_random_florest.named_steps['modelo']))
    mlflow.sklearn.log_model(
        sk_model=pipeline_random_florest,
        artifact_path="model_votacao_random_florest",
        registered_model_name="VotingRegressorPipelineRandomFlorest"
    )


with mlflow.start_run(run_name='regressao_arvore_decisao'):
    pipeline_arvore_decisao = Pipeline(steps=pipeline_completo + [
        ('modelo', DecisionTreeRegressor(**parametros_arvore_decisao))
    ])
    pipeline_arvore_decisao.fit(X=x_train.copy(), y=y_train.copy())
    previsoes = pipeline_arvore_decisao.predict(x_test.copy())
    mlflow.log_metric("rmse", math.sqrt(mean_squared_error(y_test.copy(), previsoes)))
    mlflow.sklearn.log_model(pipeline_arvore_decisao, "modelo_arvore_decisao")
    estimadores.append(("rav", pipeline_arvore_decisao.named_steps['modelo']))
    mlflow.sklearn.log_model(
        sk_model=pipeline_arvore_decisao,
        artifact_path="model_votacao_arvore_decisao",
        registered_model_name="VotingRegressorPipelineArvoreDecisao"
    )




with mlflow.start_run(run_name="voting_regressor_pipeline"):
    pipe_voting = Pipeline(steps=pipeline_completo + [
        ('modelo', VotingRegressor(estimators=estimadores, weights=[0.3, 0.3, 0.4]))
    ])
    pipe_voting.fit(x_train.copy(), y_train.copy())
    preds = pipe_voting.predict(x_test.copy())
    rmse = math.sqrt(mean_squared_error(y_test.copy(), preds))
    mlflow.log_metric("rmse", rmse)
    print(f"RMSE registrado no MLflow: {rmse:.4f}")

    exemplo_input = x_train.head(5).copy()
    exemplo_output = pipe_voting.predict(exemplo_input)

    # Inferir signature
    signature = infer_signature(exemplo_input, exemplo_output)
    mlflow.sklearn.log_model(
        sk_model=pipe_voting,
        artifact_path="model_votacao",
        registered_model_name="VotingRegressorPipeline",
        signature=signature,
        input_example=exemplo_input.head(1)  # MLflow mostrará como exemplo de input
    )
