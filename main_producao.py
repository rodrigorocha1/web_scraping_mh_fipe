import math
import pickle
import re

import mlflow
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestRegressor, VotingRegressor
from sklearn.impute import SimpleImputer
from sklearn.metrics import mean_squared_error
from sklearn.neural_network import MLPRegressor
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, OneHotEncoder, FunctionTransformer
from sklearn.tree import DecisionTreeRegressor

from src_mlops.utils.mlflow_config import configurar_mlflow

pd.set_option("display.max_rows", 200)  # linhas máximas
pd.set_option("display.max_columns", 1000)  # colunas máximas
pd.set_option("display.width", 1000)  # largura do console
pd.set_option("display.max_colwidth", 40)  # largura do conteúdo
pd.set_option("display.float_format", "{:.2f}".format)
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
# Abrindo o arquivo em modo de leitura binária (read binary)
with open('dados_completos.pkl', 'rb') as arquivo:
    dados_carregados = pickle.load(arquivo)

# Extraindo os objetos do dicionário
x_train = dados_carregados['x_train']
x_test = dados_carregados['x_test']
y_train = dados_carregados['y_train']
y_test = dados_carregados['y_test']


def extrair_motor(model_str: str):
    if pd.isna(model_str) or not isinstance(model_str, str):
        return None

    match = re.search(r'(\d\.\d)', model_str)
    if match:
        return float(match.group())
    return None


def extrair_potencia(model_str: str):
    match = re.search(r'(\d+)cv', str(model_str))
    if match:
        return int(match.group(1))
    return None


def extrair_turbo(val):
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


def extrair_transmissao(val):
    val = str(val).lower()
    if any(x in val for x in
           ['aut.', 'automático', 'automatico', 's-tronic', 'tip.', 'tiptronic', 'dsg', 'cvt', 'powershift']):
        return 'Automático'
    else:
        return 'Manual'


def extrair_turbo(val):
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


def realizar_engenharia_atributos_df(dataframe: pd.DataFrame) -> pd.DataFrame:
    dataframe['motor_cilindrada'] = dataframe['modelo'].apply(extrair_motor)
    dataframe["ano_modelo"] = dataframe["ano_modelo"].replace(32000, 2026)
    dataframe['tipo_transmissao'] = dataframe['modelo'].apply(extrair_transmissao)
    dataframe['turbo'] = dataframe['modelo'].apply(extrair_turbo)
    dataframe[colunas_categoricas] = dataframe[colunas_categoricas].astype('category')

    return dataframe


# Verificando se deu certo


num_pipeline = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='median')),
    ('scaler', StandardScaler())
])

preprocessor = ColumnTransformer(
    transformers=[
        ('num', num_pipeline, features_numericas),
        ('cat', OneHotEncoder(
            handle_unknown='ignore',
            min_frequency=30,
            sparse_output=False
        ), features_categoricas)
    ],
    verbose_feature_names_out=False
)

feature_engineering = ('feature_engineering',
                       FunctionTransformer(realizar_engenharia_atributos_df, validate=False))
preprocessador = ('preprocessor', preprocessor)
pipeline_completo = [feature_engineering, preprocessador]

MLFLOW_URI = "http://172.25.0.5:5000"

EXPERIMENT_NAME = f"modelo pronto"
configurar_mlflow(experiment_name=EXPERIMENT_NAME, tracking_uri=MLFLOW_URI)


# # Obter o experimento pelo nome
def obter_experimento(experiment_name: str):
    experiment = mlflow.get_experiment_by_name(experiment_name)
    if experiment is None:
        raise ValueError(f"Experimento '{experiment_name}' não encontrado.")

    # Buscar todos os runs do experimento, ordenando pelo mais recente
    runs = mlflow.search_runs(
        experiment_ids=[experiment.experiment_id],
        order_by=["start_time DESC"],  # mais recente primeiro
        max_results=1  # pegar apenas o run mais recente
    )
    return runs


def obter_paramentros_modelo(runs):
    if not runs.empty:

        latest_run = runs.iloc[0]

        best_params = {
            str(k).replace("params.", "").replace("best_regressor__", "").replace("regressor__", ""): v
            # convertendo k para string antes de usar replace
            for k, v in latest_run.items()
            if str(k).startswith("params.best_") or str(k).startswith("params.regressor__")
        }

        return best_params

    else:
        print("Nenhum run encontrado para este experimento.")


exp_rede_neural = obter_experimento("turing_parametros_regressao_rede_neural_v2")
exp_random_florest = obter_experimento("turing_parametros_regressao_random_florest_v2")
exp_arvore_decisao = obter_experimento("turing_parametros_regressao_arvore_de_decisao_v2")

parametros_rede_neural = obter_paramentros_modelo(exp_rede_neural)
parametros_random_florest = obter_paramentros_modelo(exp_random_florest)
parametros_arvore_decisao = obter_paramentros_modelo(exp_arvore_decisao)

import ast


def converter_parametros_mlflow(parametros: dict, tipo_modelo: str):
    """
    Converte parâmetros do MLflow (strings) para os tipos corretos esperados pelos modelos do sklearn.

    Args:
        parametros (dict): Parâmetros vindos do MLflow (geralmente strings)
        tipo_modelo (str): 'mlp', 'random_forest', 'decision_tree'

    Returns:
        dict: Parâmetros convertidos
    """
    p = parametros.copy()

    # Conversões comuns
    for chave, valor in p.items():
        if isinstance(valor, str):
            # Tenta converter para float ou int
            try:
                if '.' in valor:
                    p[chave] = float(valor)
                else:
                    p[chave] = int(valor)
            except ValueError:
                # Tenta converter para bool
                if valor.lower() == 'true':
                    p[chave] = True
                elif valor.lower() == 'false':
                    p[chave] = False
                # Tenta converter para None
                elif valor.lower() == 'none':
                    p[chave] = None
                # Tenta converter listas/tuplas usando ast.literal_eval
                else:
                    try:
                        val = ast.literal_eval(valor)
                        if isinstance(val, list):
                            # Converte listas para tupla se for hidden_layer_sizes
                            if chave == 'hidden_layer_sizes':
                                p[chave] = tuple(val)
                            else:
                                p[chave] = val
                        else:
                            p[chave] = val
                    except:
                        # Mantém string original se nada der certo
                        p[chave] = valor

    # Ajustes específicos por modelo
    if tipo_modelo == 'mlp':
        # Garantir tipos específicos
        if 'hidden_layer_sizes' in p and isinstance(p['hidden_layer_sizes'], list):
            p['hidden_layer_sizes'] = tuple(p['hidden_layer_sizes'])
        if 'early_stopping' in p:
            p['early_stopping'] = bool(p['early_stopping'])
    elif tipo_modelo in ['random_forest', 'decision_tree']:
        # Alguns parâmetros aceitam None
        for chave in ['max_depth', 'max_features', 'max_leaf_nodes', 'random_state', 'min_impurity_decrease',
                      'monotonic_cst']:
            if chave in p and p[chave] == 'None':
                p[chave] = None

    return p


parametros_rede_neural = converter_parametros_mlflow(parametros_rede_neural, 'mlp')
parametros_random_florest = converter_parametros_mlflow(parametros_random_florest, 'random_forest')
parametros_arvore_decisao = converter_parametros_mlflow(parametros_arvore_decisao, 'decision_tree')

estimadores = []

with mlflow.start_run(run_name='regressao_rede_neural'):
    pipeline_rede_neural = Pipeline(steps=pipeline_completo + [
        ('modelo', MLPRegressor(**parametros_rede_neural))
    ])
    pipeline_rede_neural.fit(X=x_train, y=y_train)
    previsoes = pipeline_rede_neural.predict(x_test)
    mlflow.log_metric("rmse", math.sqrt(mean_squared_error(y_test, previsoes)))
    mlflow.sklearn.log_model(pipeline_rede_neural, "modelo_rede_neural"),
    estimadores.append(("rrn", pipeline_rede_neural.named_steps['modelo']))
    mlflow.sklearn.log_model(
        sk_model=pipeline_rede_neural,
        artifact_path="model_votacao_rede_neural",
        registered_model_name="VotingRegressorPipelineRedeNeural"  # nome do modelo no registry
    )

with mlflow.start_run(run_name='regressao_random_florest'):
    pipeline_random_florest = Pipeline(steps=pipeline_completo + [
        ('modelo', RandomForestRegressor(**parametros_random_florest))
    ])
    pipeline_random_florest.fit(X=x_train, y=y_train)
    previsoes = pipeline_random_florest.predict(x_test)
    mlflow.log_metric("rmse", math.sqrt(mean_squared_error(y_test, previsoes)))
    mlflow.sklearn.log_model(pipeline_random_florest, "modelo_random_florest"),
    estimadores.append(("rrf", pipeline_random_florest.named_steps['modelo']))
    mlflow.sklearn.log_model(
        sk_model=pipeline_random_florest,
        artifact_path="model_votacao_random_florest",
        registered_model_name="VotingRegressorPipelineRandomFlorest"  # nome do modelo no registry
    )

with mlflow.start_run(run_name='regressao_arvore_decisao'):
    pipeline_arvore_decisao = Pipeline(steps=pipeline_completo + [
        ('modelo', DecisionTreeRegressor(**parametros_arvore_decisao))])
    pipeline_arvore_decisao.fit(X=x_train, y=y_train)
    previsoes = pipeline_arvore_decisao.predict(x_test)
    mlflow.log_metric("rmse", math.sqrt(mean_squared_error(y_test, previsoes)))
    mlflow.sklearn.log_model(pipeline_arvore_decisao, "modelo_random_florest"),
    estimadores.append(("rav", pipeline_arvore_decisao.named_steps['modelo']))

    mlflow.sklearn.log_model(
        sk_model=pipeline_arvore_decisao,
        artifact_path="model_votacao_arvore_decisao",
        registered_model_name="VotingRegressorPipelineArvoreDecisao"  # nome do modelo no registry
    )

with mlflow.start_run(run_name="voting_regressor_pipeline"):
    pipe_voting = Pipeline(steps=pipeline_completo + [
        ('modelo', VotingRegressor(
            estimators=estimadores,
            weights=[0.3, 0.3, 0.4]  # você pode ajustar baseado nos RMSE individuais
        ))
    ])
    pipe_voting.fit(x_train, y_train)

    # Faz predições
    preds = pipe_voting.predict(x_test)

    # Loga a métrica RMSE
    rmse = math.sqrt(mean_squared_error(y_test, preds))
    mlflow.log_metric("rmse", rmse)
    print(f"RMSE registrado no MLflow: {rmse:.4f}")

    # Loga e registra o modelo no MLflow Model Registry
    mlflow.sklearn.log_model(
        sk_model=pipe_voting,
        artifact_path="model_votacao",
        registered_model_name="VotingRegressorPipeline"  # nome do modelo no registry
    )