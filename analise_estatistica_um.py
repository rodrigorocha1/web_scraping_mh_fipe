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
        input_file='dados/resultados_validacao_cruzada/regressao_arvore_de_decisao/resultado_validacao_cruzada_regressao_arvore_de_decisao.jsonl',
        nome_modelo='regressao_arvore_de_decisao'

    )

    df_resultado_regressao_linear = abrir_dataframe_residuos(
        input_file='dados/resultados_validacao_cruzada/regressao_linear/resultado_validacao_cruzada_regressao_linear.jsonl',
        nome_modelo='regressao_linear'
    )

    df_resultado_elastic_net = abrir_dataframe_residuos(
        input_file='dados/resultados_validacao_cruzada/regressao_elastic_net/resultado_validacao_cruzada_regressao_elastic_net.jsonl',
        nome_modelo='regressao_elastic_net'

    )

    df_resultado_random_florest = abrir_dataframe_residuos(
        input_file='dados/resultados_validacao_cruzada/regressao_random_florest/resultado_validacao_cruzada_regressao_random_florest.jsonl',
        nome_modelo='regressao_random_florest'
    )
    df_resultado_lasso = abrir_dataframe_residuos(
        input_file='dados/resultados_validacao_cruzada/regressao_linear_lasso/resultado_validacao_cruzada_regressao_linear_lasso.jsonl',
        nome_modelo='regressao_linear_lasso'
    )

    df_resultado_ridge = abrir_dataframe_residuos(
        input_file='dados/resultados_validacao_cruzada/regressao_linear_ridge/resultado_validacao_cruzada_regressao_linear_ridge.jsonl',
        nome_modelo='regressao_linear_ridge'
    )

    df_resultado_svr = abrir_dataframe_residuos(
        input_file='dados/resultados_validacao_cruzada/regressao_s_v_r/resultado_validacao_cruzada_regressao_s_v_r.jsonl',
        nome_modelo='regressao_s_v_r'
    )

    df_resultado_rede_neural = abrir_dataframe_residuos(
        input_file='dados/resultados_validacao_cruzada/regressao_rede_neural/resultado_validacao_cruzada_regressao_rede_neural.jsonl',
        nome_modelo='regressao_rede_neural'
    )
    df_resultado_random_florest.drop(columns=['iteracao'], inplace=True)
    df_resultado_svr.drop(columns=['iteracao'], inplace=True)
    df_resultado_rede_neural.drop(columns=['iteracao'], inplace=True)
    df_resultado_lasso.drop(columns=['iteracao'], inplace=True)
    df_resultado_ridge.drop(columns=['iteracao'], inplace=True)
    df_resultado_elastic_net.drop(columns='iteracao', inplace=True)
    df_resultado_regressao_linear.drop(columns='iteracao', inplace=True)

    df_completo = pd.concat(
        [df_resultado_arvore_decisao, df_resultado_random_florest, df_resultado_svr, df_resultado_rede_neural,
         df_resultado_elastic_net, df_resultado_lasso, df_resultado_ridge, df_resultado_regressao_linear], axis=1)
    colunas_residuos = [
        'residuos_totais_regressao_arvore_de_decisao',
        'residuos_totais_regressao_random_florest',
        'residuos_totais_regressao_s_v_r',
        'residuos_totais_regressao_rede_neural',
        'residuos_totais_regressao_elastic_net',
        'residuos_totais_regressao_linear_lasso',
        'residuos_totais_regressao_linear_ridge',
        'residuos_totais_regressao_linear'
    ]

    print(df_completo.head())

    for col in colunas_residuos:
        df_completo[col] = pd.to_numeric(df_completo[col])

    df_completo.to_csv('residuos.csv', index=False, sep='|')

    df_ciclo_zero = df_completo[df_completo['iteracao'] == 0]


def extrair_rmse(input_file: str):
    df_bruto = abrir_dataframe_bruto(input_file=input_file)


    df_bruto = df_bruto[['nome_modelo', 'iteracao', 'mean_scores']]

    print(df_bruto)
    df_bruto['mean_test_rmse'] = df_bruto['mean_scores'].apply(lambda d: d.get('mean_test_rmse'))
    df_bruto['mean_train_rmse'] = df_bruto['mean_scores'].apply(lambda d: d.get('mean_train_mse'))
    df_bruto.drop('mean_scores', axis=1, inplace=True)




    return df_bruto


def extrair_rmse_completa():
    dataframes_rmse_rede_neural = extrair_rmse(
        input_file='dados/resultados_validacao_cruzada/regressao_rede_neural/resultado_validacao_cruzada_regressao_rede_neural.jsonl')
    dataframes_rmse_arvore_decisao = extrair_rmse(
        input_file='dados/resultados_validacao_cruzada/regressao_arvore_de_decisao/resultado_validacao_cruzada_regressao_arvore_de_decisao.jsonl')

    dataframes_rmse_random_florest = extrair_rmse(
        input_file='dados/resultados_validacao_cruzada/regressao_random_florest/resultado_validacao_cruzada_regressao_random_florest.jsonl')
    dataframes_rmse_regressao_svr = extrair_rmse(
        input_file='dados/resultados_validacao_cruzada/regressao_s_v_r/resultado_validacao_cruzada_regressao_s_v_r.jsonl')

    dataframe_rmse_lasso = extrair_rmse(
        input_file='dados/resultados_validacao_cruzada/regressao_linear_lasso/resultado_validacao_cruzada_regressao_linear_lasso.jsonl'
    )

    dataframe_rmse_linear = extrair_rmse(
        input_file='dados/resultados_validacao_cruzada/regressao_linear/resultado_validacao_cruzada_regressao_linear.jsonl'
    )

    dataframe_rmse_ridge = extrair_rmse(
        input_file='dados/resultados_validacao_cruzada/regressao_linear_ridge/resultado_validacao_cruzada_regressao_linear_ridge.jsonl'
    )

    dataframa_rmse_elastic_net = extrair_rmse(
        input_file='dados/resultados_validacao_cruzada/regressao_elastic_net/resultado_validacao_cruzada_regressao_elastic_net.jsonl'
    )
    dataframe_rmse_completo = pd.concat(
        [dataframes_rmse_rede_neural, dataframes_rmse_arvore_decisao, dataframes_rmse_random_florest,
         dataframes_rmse_regressao_svr, dataframe_rmse_lasso, dataframe_rmse_linear, dataframe_rmse_ridge,
         dataframa_rmse_elastic_net, ], axis=0)
    print(dataframe_rmse_completo.head())
    dataframe_rmse_completo.to_csv('rmse.csv', index=False, sep='|')

extrair_rmse_completa()
