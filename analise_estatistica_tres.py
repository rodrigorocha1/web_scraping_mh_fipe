import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import norm
import scipy.stats as stats
from matplotlib.backends.backend_pdf import PdfPages
import numpy as np
import os

pd.set_option('display.max_columns', None)
pd.set_option('display.max_rows', None)
pd.set_option('display.width', None)
pd.set_option('display.max_colwidth', None)



pd.set_option('display.precision', 4)

df = pd.read_csv(
    "residuos.csv",
    sep="|"
)

print(df.head())
print(df.columns)

erro_df = pd.DataFrame({
    "arvore_decisao": df["residuos_totais_arvore_decisao"].abs(),
    "random_florest": df["residuos_totais_random_florest"].abs(),
    "regressao_svr": df["residuos_totais_regressao_svr"].abs(),
    "rede_neural": df["residuos_totais_rede_neural"].abs(),
})

erro_df.head()

from scipy.stats import friedmanchisquare

stat, p_value = friedmanchisquare(
    erro_df["arvore_decisao"],
    erro_df["random_florest"],
    erro_df["rede_neural"],
    erro_df["regressao_svr"]
)

print(f"Friedman chi-square: {stat:.4f}")
print(f"p-value: {p_value:.8f}")



import scikit_posthocs as sp

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

