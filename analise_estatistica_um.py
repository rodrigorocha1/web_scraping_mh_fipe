import json

import pandas as pd

pd.set_option('display.max_columns', None)
pd.set_option('display.max_rows', None)
pd.set_option('display.width', None)
pd.set_option('display.max_colwidth', None)


def abrir_dataframe_bruto(input_file):
    lista_dicionarios = []
    with open(input_file, 'r', encoding='utf-8') as arquivo:
        for linha in arquivo:
            dados_iteracao = json.loads(linha)
            lista_dicionarios.append(dados_iteracao)

    df_bruto = pd.DataFrame(lista_dicionarios)
    return df_bruto


def abrir_dataframe_residuos(
        input_file: str,
        nome_modelo: str,
) -> pd.DataFrame:
    df_bruto = abrir_dataframe_bruto(input_file)

    df_mean_scores = df_bruto[['nome_modelo', 'iteracao', 'residuos_totais']].explode('residuos_totais')

    df_mean_scores = df_mean_scores[df_mean_scores['nome_modelo'] == nome_modelo]
    df_mean_scores.rename(columns={'residuos_totais': f'residuos_totais_{nome_modelo}'}, inplace=True)
    df_mean_scores.drop(columns=['nome_modelo'], inplace=True)
    return df_mean_scores


def gerar_dataframe_residuos():
    df_resultado_arvore_decisao = abrir_dataframe_residuos(
        input_file='dados/resultados_validacao_cruzada/arvore_decisao/resultado_validacao_cruzada_arvore_decisao.jsonl',
        nome_modelo='arvore_decisao'

    )

    df_resultado_random_florest = abrir_dataframe_residuos(
        input_file='dados/resultados_validacao_cruzada/random_florest/resultado_validacao_cruzada_random_florest.jsonl',
        nome_modelo='random_florest'
    )

    df_resultado_svr = abrir_dataframe_residuos(
        input_file='dados/resultados_validacao_cruzada/regressao_svr/resultado_validacao_cruzada_regressao_svr.jsonl',
        nome_modelo='regressao_svr'
    )

    df_resultado_rede_neural = abrir_dataframe_residuos(
        input_file='dados/resultados_validacao_cruzada/rede_neural/resultado_validacao_cruzada_rede_neural.jsonl',
        nome_modelo='rede_neural'
    )
    df_resultado_random_florest.drop(columns=['iteracao'], inplace=True)
    df_resultado_svr.drop(columns=['iteracao'], inplace=True)
    df_resultado_rede_neural.drop(columns=['iteracao'], inplace=True)

    df_completo = pd.concat(
        [df_resultado_arvore_decisao, df_resultado_random_florest, df_resultado_svr, df_resultado_rede_neural], axis=1)
    colunas_residuos = [
        'residuos_totais_arvore_decisao',
        'residuos_totais_random_florest',
        'residuos_totais_regressao_svr',
        'residuos_totais_rede_neural'
    ]

    for col in colunas_residuos:
        df_completo[col] = pd.to_numeric(df_completo[col])

    df_completo.to_csv('residuos.csv', index=False, sep='|')

    df_ciclo_zero = df_completo[df_completo['iteracao'] == 0]


def extrair_rmse(input_file: str):
    df_bruto = abrir_dataframe_bruto(input_file=input_file)
    print(df_bruto.head())

    df_bruto = df_bruto[['nome_modelo', 'iteracao', 'mean_scores']]


    df_bruto['mean_test_rmse'] = df_bruto['mean_scores'].apply(lambda d: d.get('mean_test_rmse'))
    df_bruto.drop('mean_scores', axis=1, inplace=True)

    return df_bruto


def extrair_rmse_completa():
    dataframes_rmse_rede_neural = extrair_rmse(
        input_file='dados/resultados_validacao_cruzada/rede_neural/resultado_validacao_cruzada_rede_neural.jsonl')
    dataframes_rmse_arvore_decisao = extrair_rmse(
        input_file='dados/resultados_validacao_cruzada/arvore_decisao/resultado_validacao_cruzada_arvore_decisao.jsonl')

    dataframes_rmse_random_florest = extrair_rmse(
        input_file='dados/resultados_validacao_cruzada/random_florest/resultado_validacao_cruzada_random_florest.jsonl')
    dataframes_rmse_regressao_svr = extrair_rmse(
        input_file='dados/resultados_validacao_cruzada/regressao_svr/resultado_validacao_cruzada_regressao_svr.jsonl')
    dataframe_rmse_completo = pd.concat(
        [dataframes_rmse_rede_neural, dataframes_rmse_arvore_decisao, dataframes_rmse_random_florest,
         dataframes_rmse_regressao_svr], axis=0)

    dataframe_rmse_completo.to_csv('rmse.csv', index=False, sep='|')

extrair_rmse_completa()
