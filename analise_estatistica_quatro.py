import matplotlib.pyplot as plt
import pandas as pd

from scipy.stats import friedmanchisquare
import scikit_posthocs as sp

pd.set_option('display.max_columns', None)
pd.set_option('display.max_rows', None)
pd.set_option('display.width', None)
pd.set_option('display.max_colwidth', None)

pd.set_option('display.precision', 4)

df = pd.read_csv(
    "rmse.csv",
    sep="|"
)

print(df.head())
print(df.columns)
df_pivot = df.pivot(
    index='iteracao',
    columns='nome_modelo',
    values='mean_test_rmse'
)
df_pivot.columns.name = None
df_pivot.index.name = None
print(df_pivot)

erro_df = df_pivot.copy()

print(erro_df.head())

stat, p_value = friedmanchisquare(
    erro_df["arvore_decisao"],
    erro_df["random_florest"],
    erro_df["rede_neural"],
    erro_df["regressao_svr"]
)

print(f"Friedman chi-square: {stat:.4f}")
print(f"p-value: {p_value:.8f}")



nemenyi = sp.posthoc_nemenyi_friedman(erro_df)

print(nemenyi)

ranks = erro_df.rank(axis=1, method="average")

mean_ranks = ranks.mean().sort_values()
print(mean_ranks)

from scikit_posthocs import critical_difference_diagram

plt.figure(figsize=(10, 3))
critical_difference_diagram(
    mean_ranks,
    nemenyi,

)

plt.title("Critical Difference Diagram – Friedman + Nemenyi")
plt.show()

from scipy.stats import studentized_range
import numpy as np

alpha = 0.05
k = erro_df.shape[1]   # número de modelos
N = erro_df.shape[0]   # número de iterações

q_alpha = studentized_range.ppf(1 - alpha, k, np.inf)

CD = q_alpha * np.sqrt((k * (k + 1)) / (6 * N))


import matplotlib.pyplot as plt

plt.figure(figsize=(10, 3))

# eixo principal
plt.hlines(1, xmin=mean_ranks.min() - 0.2, xmax=mean_ranks.max() + 0.2)
plt.yticks([])
plt.xlabel("Rank médio (menor = melhor)")
plt.title("Critical Difference Diagram – Friedman + Nemenyi")

# pontos dos modelos
for i, (model, rank) in enumerate(mean_ranks.items()):
    plt.plot(rank, 1, 'o')
    plt.text(rank, 0.95, f"{model}\n({rank:.2f})",
             ha='center', va='top')

# barra da CD
cd_x_start = mean_ranks.max() - CD
cd_x_end = mean_ranks.max()

plt.plot([cd_x_start, cd_x_end], [1.15, 1.15], lw=2)
plt.text((cd_x_start + cd_x_end)/2, 1.18, f"CD = {CD:.2f}",
         ha='center')

plt.ylim(0.85, 1.25)
plt.show()