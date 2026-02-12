import io
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import mlflow
import scikit_posthocs as sp

from datetime import datetime
from mlflow.tracking import MlflowClient
from matplotlib.backends.backend_pdf import PdfPages
from scipy import stats
from scipy.stats import norm, friedmanchisquare, studentized_range
from scikit_posthocs import critical_difference_diagram
from statsmodels.stats.multicomp import MultiComparison


# ==========================================================
# CONFIGURAÇÕES
# ==========================================================

MLFLOW_URI = "http://172.25.0.5:5000"
mlflow.set_tracking_uri(MLFLOW_URI)
mlflow.set_experiment("analise_estatistica_modelos_v2")

mlflow_client = MlflowClient(tracking_uri=MLFLOW_URI)

pd.set_option("display.float_format", "{:.4f}".format)


# ==========================================================
# FUNÇÕES
# ==========================================================

def executar_testes_normalidade(df):
    resultados = []

    for modelo in df['experiment_name'].unique():
        dados = df.loc[
            df['experiment_name'] == modelo,
            'mean_test_rmse'
        ].dropna()

        if len(dados) < 3:
            continue

        stat_sw, p_sw = stats.shapiro(dados)
        stat_k2, p_k2 = stats.normaltest(dados)

        media, desvio = dados.mean(), dados.std(ddof=1)
        stat_ks, p_ks = stats.kstest(dados, 'norm', args=(media, desvio))

        resultados.append({
            'modelo': modelo,
            'shapiro_p': p_sw,
            'k2_p': p_k2,
            'ks_p': p_ks
        })

    return pd.DataFrame(resultados)


def gerar_distribuicao_rmse(df):

    modelos = df['experiment_name'].unique()
    n_cols = 2
    n_rows = int(np.ceil(len(modelos) / n_cols))

    fig, axes = plt.subplots(n_rows, n_cols, figsize=(12, 4 * n_rows))
    axes = axes.flatten()

    fig.suptitle('Distribuição do RMSE por Modelo', fontsize=16)

    for i, modelo in enumerate(modelos):
        ax = axes[i]

        dados = df.loc[
            df['experiment_name'] == modelo,
            'mean_test_rmse'
        ].dropna()

        sns.histplot(dados, kde=True, stat="density", ax=ax)

        mu, std = norm.fit(dados)
        x = np.linspace(dados.min(), dados.max(), 100)
        ax.plot(x, norm.pdf(x, mu, std), lw=2)

        ax.set_title(f'{modelo}\nμ={mu:.2f} | σ={std:.2f}')
        ax.set_xlabel('RMSE')

    # remove eixos extras
    for j in range(i + 1, len(axes)):
        fig.delaxes(axes[j])

    plt.tight_layout(rect=[0, 0.03, 1, 0.95])

    # ✅ PNG
    mlflow.log_figure(fig, "graficos/distribuicao_normal.png")

    # ✅ PDF (mesma figura, outro formato)
    mlflow.log_figure(fig, "graficos/distribuicao_normal.pdf")

    plt.close(fig)



def executar_tukey(df):
    df_long = df[['experiment_name', 'mean_test_rmse']].dropna()

    mc = MultiComparison(df_long['mean_test_rmse'], df_long['experiment_name'])
    tukey = mc.tukeyhsd()

    fig = tukey.plot_simultaneous(figsize=(10, 6))
    plt.title("Tukey HSD – Comparação de RMSE")

    mlflow.log_figure(fig, "graficos/tukey_rmse.png")
    plt.close(fig)

    return tukey


# ==========================================================
# COLETAR DADOS DO MLFLOW
# ==========================================================

lista_experimento = [
    'validacao_cruzada_regressao_elastic_net_v2',
    'validacao_cruzada_regressao_linear_ridge_v2',
    'validacao_cruzada_regressao_linear_lasso_v2',
    'validacao_cruzada_regressao_linear_v2',
    'validacao_cruzada_regressao_random_florest_v2',
    'validacao_cruzada_regressao_rede_neural_v2',
    'validacao_cruzada_regressao_s_v_r_v2',
    'validacao_cruzada_regressao_arvore_de_decisao_v2'
]

dfs = []

for nome in lista_experimento:
    experiment = mlflow_client.get_experiment_by_name(nome)
    if experiment is None:
        continue

    runs = mlflow_client.search_runs(
        experiment_ids=[experiment.experiment_id]
    )

    dados = []

    for run in runs:
        metrics = run.data.metrics

        dados.append({
            "experiment_name": " ".join(
                experiment.name.replace('v2', '').split('_')[2:]
            ).strip(),
            "run_name": run.data.tags.get("mlflow.runName"),
            "mean_test_rmse": metrics.get("mean_test_rmse"),
        })

    dfs.append(pd.DataFrame(dados))

df_final = pd.concat(dfs, ignore_index=True)


# ==========================================================
# RUN DE ANÁLISE ESTATÍSTICA
# ==========================================================

with mlflow.start_run(
        run_name=f"analise_estatistica_{datetime.now().strftime('%Y%m%d_%H%M')}"):

    mlflow.log_param("alpha", 0.05)
    mlflow.log_param("qtd_modelos", df_final["experiment_name"].nunique())
    mlflow.log_param("qtd_execucoes", df_final["run_name"].nunique())

    mlflow.log_dict(
        df_final.to_dict(orient="records"),
        artifact_file="dados/df_final.json"
    )

    # ================= NORMALIDADE =================
    df_normalidade = executar_testes_normalidade(df_final)

    mlflow.log_dict(
        df_normalidade.to_dict(orient="records"),
        artifact_file="estatistica/teste_normalidade.json"
    )

    # ================= DISTRIBUIÇÃO =================
    gerar_distribuicao_rmse(df_final)

    # ================= TUKEY =================
    executar_tukey(df_final)

    # ================= FRIEDMAN =================
    erro_df = df_final.pivot(
        index='run_name',
        columns='experiment_name',
        values='mean_test_rmse'
    ).dropna()

    stat, p_value = friedmanchisquare(
        *[erro_df[col] for col in erro_df.columns]
    )

    mlflow.log_metric("friedman_statistic", float(stat))
    mlflow.log_metric("friedman_p_value", float(p_value))

    # ================= NEMENYI =================
    nemenyi = sp.posthoc_nemenyi_friedman(erro_df)

    mlflow.log_dict(
        nemenyi.to_dict(),
        artifact_file="estatistica/nemenyi_matrix.json"
    )

    # ================= RANKS =================
    ranks = erro_df.rank(axis=1, method="average")
    mean_ranks = ranks.mean().sort_values()

    mlflow.log_dict(
        mean_ranks.to_dict(),
        artifact_file="estatistica/mean_ranks.json"
    )

    mlflow.log_metric("melhor_rank", float(mean_ranks.min()))

    fig = plt.figure(figsize=(12, 5))
    ranks.boxplot(rot=30, grid=False)
    plt.ylabel("Rank (menor = melhor)")
    plt.title("Distribuição dos Ranks – Friedman")
    mlflow.log_figure(fig, "graficos/friedman_ranks_boxplot.png")
    plt.close(fig)

    # ================= CRITICAL DIFFERENCE =================
    alpha = 0.05
    k = erro_df.shape[1]
    N = erro_df.shape[0]

    q_alpha = studentized_range.ppf(1 - alpha, k, np.inf)
    CD = q_alpha * np.sqrt((k * (k + 1)) / (6 * N))

    mlflow.log_metric("critical_difference", float(CD))

    fig = plt.figure(figsize=(10, 3))
    critical_difference_diagram(mean_ranks, nemenyi)
    plt.title("Critical Difference Diagram")
    mlflow.log_figure(fig, "graficos/cd_diagram.png")
    plt.close(fig)

print("✅ Análise completa registrada no MLflow sem salvar arquivos locais.")
