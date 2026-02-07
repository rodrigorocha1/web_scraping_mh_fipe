import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

from scipy.stats import friedmanchisquare, studentized_range
import scikit_posthocs as sp
from scikit_posthocs import critical_difference_diagram

# =========================
# CONFIGURAÇÕES DO PANDAS
# =========================
pd.set_option('display.max_columns', None)
pd.set_option('display.max_rows', None)
pd.set_option('display.width', None)
pd.set_option('display.max_colwidth', None)
pd.set_option('display.precision', 4)

# =========================
# LEITURA DOS DADOS
# =========================
df = pd.read_csv(
    "rmse.csv",
    sep="|"
)

print(df.head())
print(df.columns)

# =========================
# PIVOT
# =========================
df_pivot = df.pivot(
    index='iteracao',
    columns='nome_modelo',
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
    erro_df["regressao_arvore_de_decisao"],
    erro_df["regressao_elastic_net"],
    erro_df["regressao_linear"],
    erro_df["regressao_linear_lasso"],
    erro_df["regressao_linear_ridge"],
    erro_df["regressao_random_florest"],
    erro_df["regressao_rede_neural"],
    erro_df["regressao_s_v_r"],
)

print(f"Friedman chi-square: {stat:.4f}")
print(f"p-value: {p_value:.8f}")



# =========================
# NEMENYI POST-HOC
# =========================
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