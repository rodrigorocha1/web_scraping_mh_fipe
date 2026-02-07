import pandas as pd
import matplotlib.pyplot as plt
import os

# carregar e tratar
df_raw = pd.read_csv("rmse.csv")
df = df_raw.iloc[:,0].str.split("|", expand=True)
df.columns = ["modelo", "iteracao", "rmse"]
df["iteracao"] = df["iteracao"].astype(int)
df["rmse"] = df["rmse"].astype(float)

out_dir = "fig/graficos_overfitting"
os.makedirs(out_dir, exist_ok=True)

paths = []

for modelo, g in df.groupby("modelo"):
    g = g.sort_values("iteracao")
    plt.figure()
    plt.plot(g["iteracao"], g["rmse"])
    plt.xlabel("Iteração")
    plt.ylabel("RMSE (validação)")
    plt.title(f"Curva de erro – {modelo}")
    path = f"{out_dir}/{modelo}_overfitting.png"
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    paths.append(path)

paths
