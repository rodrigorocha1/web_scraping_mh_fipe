import os

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib.backends.backend_pdf import PdfPages
from mlflow.tracking import MlflowClient
from scikit_posthocs import critical_difference_diagram
from scipy import stats
from scipy.stats import norm, friedmanchisquare, studentized_range

pd.set_option("display.max_rows", 200)
pd.set_option("display.max_columns", 1000)
pd.set_option("display.width", 1000)
pd.set_option("display.max_colwidth", 40)
pd.set_option("display.float_format", "{:.2f}".format)

MLFLOW_URI = "http://172.25.0.5:5000"
mlflow_client = MlflowClient(tracking_uri=MLFLOW_URI)


def executar_testes_normalidade(df):
    resultados = []

    modelos = df['experiment_name'].unique()

    for modelo in modelos:
        dados = df.loc[
            df['experiment_name'] == modelo,
            'mean_test_rmse'
        ].dropna()

        if len(dados) < 3:
            print(f"⚠️ Modelo {modelo} tem poucos dados, pulando...")
            continue

        stat_sw, p_sw = stats.shapiro(dados)
        stat_k2, p_k2 = stats.normaltest(dados)

        media, desvio = dados.mean(), dados.std(ddof=1)
        stat_ks, p_ks = stats.kstest(dados, 'norm', args=(media, desvio))

        resultados.append({
            'Modelo': modelo,
            'Shapiro_Stat': stat_sw,
            'Shapiro_P_Value': p_sw,
            'K2_Stat': stat_k2,
            'K2_P_Value': p_k2,
            'KS_Stat': stat_ks,
            'KS_P_Value': p_ks
        })

    return pd.DataFrame(resultados)


def processar_e_salvar_resultados(df_resultados, alpha=0.05):
    df_resultados['Shapiro_Normal'] = df_resultados['Shapiro_P_Value'] > alpha
    df_resultados['K2_Normal'] = df_resultados['K2_P_Value'] > alpha
    df_resultados['KS_Normal'] = df_resultados['KS_P_Value'] > alpha

    return df_resultados


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

dfs = []  # 🔹 lista para acumular os DataFrames

for nome_experimento in lista_experimento:
    experiment = mlflow_client.get_experiment_by_name(nome_experimento)
    experiment_id = experiment.experiment_id

    runs = mlflow_client.search_runs(
        experiment_ids=[experiment_id],
        order_by=["attributes.start_time DESC"]
    )

    dados = []

    for run in runs:
        metrics = run.data.metrics

        dados.append({
            "experiment_name": " ".join(experiment.name.replace('v2', '').split('_')[2:]).strip(),
            "run_name": run.data.tags.get("mlflow.runName"),
            "mean_fit_time": metrics.get("mean_fit_time"),
            "mean_score_time": metrics.get("mean_score_time"),
            "mean_test_mae": metrics.get("mean_test_mae"),
            "mean_test_mse": metrics.get("mean_test_mse"),
            "mean_test_rmse": metrics.get("mean_test_rmse"),
            "mean_test_r2": metrics.get("mean_test_r2"),
        })

    df_metrics = pd.DataFrame(dados)
    dfs.append(df_metrics)  # 🔹 acumula


def gerar_graficos_distribuicao_rmse(df, modelos, labels, nome_pdf, nome_img):
    os.makedirs(os.path.dirname(nome_pdf) or '.', exist_ok=True)

    n_modelos = len(modelos)
    n_cols = 2
    n_rows = int(np.ceil(n_modelos / n_cols))

    with PdfPages(nome_pdf) as pdf:
        fig, axes = plt.subplots(n_rows, n_cols, figsize=(12, 4 * n_rows))
        axes = axes.flatten()

        fig.suptitle('Distribuição do RMSE por Modelo', fontsize=16)

        for i, (modelo, label) in enumerate(zip(modelos, labels)):
            ax = axes[i]


            dados = df.loc[
                df['experiment_name'] == modelo,
                'mean_test_rmse'
            ].dropna()

            sns.histplot(dados, kde=True, stat="density", ax=ax)

            mu, std = norm.fit(dados)
            x = np.linspace(dados.min(), dados.max(), 100)
            ax.plot(x, norm.pdf(x, mu, std), lw=2)

            ax.set_title(f'{label}\nμ={mu:.2f} | σ={std:.2f}')
            ax.set_xlabel('RMSE')

        for j in range(i + 1, len(axes)):
            fig.delaxes(axes[j])

        plt.tight_layout(rect=[0, 0.03, 1, 0.95])
        pdf.savefig(fig)
        fig.savefig(f'fig/{nome_img}', dpi=300, bbox_inches="tight")
        plt.close(fig)

    print(f"\n📄 PDF salvo: {nome_pdf}")
    print(f"🖼️ Imagem salva: {nome_img}")

def executar_testes_multicomp_rmse(df, arquivo_figura):
    from statsmodels.stats.multicomp import MultiComparison

    df_long = df[['experiment_name', 'mean_test_rmse']].dropna()

    mc = MultiComparison(df_long['mean_test_rmse'], df_long['experiment_name'])
    tukey = mc.tukeyhsd()

    print("\n📊 Resultado Tukey HSD")
    print(tukey)

    fig = tukey.plot_simultaneous(figsize=(10, 6))
    plt.title("Tukey HSD – Comparação de RMSE entre Modelos")
    plt.savefig(f'fig/{arquivo_figura}', dpi=300, bbox_inches="tight")
    plt.close()

    print(f"🖼️ Gráfico Tukey salvo em: fig/{arquivo_figura}")
df_final: pd.DataFrame = pd.concat(dfs, ignore_index=True)
print(df_final)
# df_processado = executar_testes_normalidade(df_final)
#
# df_processado = processar_e_salvar_resultados(df_resultados=df_processado, )
#
#
#
# labels = df_final['experiment_name'].unique()
# modelos = df_final['experiment_name'].unique()
# gerar_graficos_distribuicao_rmse(
#     df_final,
#     modelos=modelos,
#     labels=labels,
#     nome_img='distribuicao_normal.png',
#     nome_pdf='distribuicao_normal.pdf'
# )
#
# executar_testes_multicomp_rmse(
#         df_final,
#         "tukey_rmse.png"
#     )

df_pivot = df_final.pivot(
    index='run_name',
    columns='experiment_name',
    values='mean_test_rmse'
)

df_pivot.columns.name = None
df_pivot.index.name = None

erro_df = df_pivot.copy()

print(erro_df.head())


# =========================
# TESTE DE FRIEDMAN
# =========================
stat, p_value = friedmanchisquare(
    erro_df["regressao arvore de decisao"],
    erro_df["regressao elastic net"],
    erro_df["regressao linear"],
    erro_df["regressao linear lasso"],
    erro_df["regressao linear ridge"],
    erro_df["regressao random florest"],
    erro_df["regressao rede neural"],
    erro_df["regressao s v r"],
)

print(f"Friedman chi-square: {stat:.4f}")
print(f"p-value: {p_value:.8f}")

import scikit_posthocs as sp

nemenyi = sp.posthoc_nemenyi_friedman(erro_df)
print(nemenyi)

# =========================
# RANKS
# =========================
ranks = erro_df.rank(axis=1, method="average")
mean_ranks = ranks.mean().sort_values()

print(mean_ranks)
plt.figure(figsize=(12, 5))

ranks.boxplot(
    rot=30,
    grid=False
)

plt.ylabel("Rank (menor = melhor)")
plt.title("Teste de Friedman – Distribuição dos Ranks por Modelo")

plt.tight_layout()
plt.savefig(
    "fig/friedman_ranks_boxplot.png",
    dpi=300,
    bbox_inches="tight"
)
plt.close()




# =========================
# CD DIAGRAM (scikit-posthocs)
# =========================
plt.figure(figsize=(10, 3))

critical_difference_diagram(
    mean_ranks,
    nemenyi
)

plt.title("Critical Difference Diagram – Friedman + Nemenyi")

plt.savefig(
    "fig/cd_diagram_scikit_posthocs.png",
    dpi=300,
    bbox_inches="tight"
)

plt.close()

# =========================
# CÁLCULO DA CRITICAL DIFFERENCE (MANUAL)
# =========================
alpha = 0.05
k = erro_df.shape[1]   # número de modelos
N = erro_df.shape[0]   # número de iterações

q_alpha = studentized_range.ppf(1 - alpha, k, np.inf)

CD = q_alpha * np.sqrt((k * (k + 1)) / (6 * N))

print(f"Critical Difference (CD): {CD:.4f}")

# =========================
# CD DIAGRAM MANUAL
# =========================

labels_curto = {
    "regressao_arvore_de_decisao": "Árvore",
    "regressao_elastic_net": "ElasticNet",
    "regressao_linear": "Linear",
    "regressao_linear_lasso": "Lasso",
    "regressao_linear_ridge": "Ridge",
    "regressao_random_florest": "RandomForest",
    "regressao_rede_neural": "Rede Neural",
    "regressao_s_v_r": "SVR",
}

plt.figure(figsize=(12, 3))

# linha base
plt.hlines(
    y=1,
    xmin=mean_ranks.min() - 0.3,
    xmax=mean_ranks.max() + 0.3
)

plt.yticks([])
plt.xlabel("Rank médio (menor = melhor)")
plt.title("Critical Difference Diagram – Friedman + Nemenyi")

# pontos dos modelos + labels curtos
for model, rank in mean_ranks.items():
    label = labels_curto.get(model, model)

    plt.plot(rank, 1, 'o')
    plt.text(
        rank,
        0.93,
        f"{label}\n({rank:.2f})",
        ha='center',
        va='top',
        fontsize=9
    )

# barra da CD
cd_x_start = mean_ranks.max() - CD
cd_x_end = mean_ranks.max()

plt.plot(
    [cd_x_start, cd_x_end],
    [1.15, 1.15],
    lw=2
)

plt.text(
    (cd_x_start + cd_x_end) / 2,
    1.18,
    f"CD = {CD:.2f}",
    ha='center',
    fontsize=10
)

plt.ylim(0.85, 1.28)

plt.savefig(
    "fig/cd_diagram_manual.png",
    dpi=300,
    bbox_inches="tight"
)

plt.close()