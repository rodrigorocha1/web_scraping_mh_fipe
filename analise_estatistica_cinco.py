import os

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import scipy.stats as stats
import seaborn as sns
from matplotlib.backends.backend_pdf import PdfPages
from scipy.stats import norm


# =========================================================
# Configurações gerais
# =========================================================
def configurar_pandas():
    pd.set_option('display.max_columns', None)
    pd.set_option('display.max_rows', None)
    pd.set_option('display.width', 1000)
    pd.set_option('display.max_colwidth', 50)
    pd.set_option('display.precision', 4)


# =========================================================
# Carregamento dos dados
# =========================================================
def carregar_dados(caminho_arquivo, separador='|'):
    df = pd.read_csv(caminho_arquivo, sep=separador)

    print("\n📊 Dataset carregado")
    print(df.head())
    print("\nℹ️ Info:")
    print(df.info())

    print(f"\n🔁 Iterações únicas: {df['iteracao'].nunique()}")
    return df


# =========================================================
# Testes de normalidade (RMSE por modelo)
# =========================================================
def executar_testes_normalidade(df, modelos):
    resultados = []

    for modelo in modelos:
        dados = df.loc[
            df['nome_modelo'] == modelo,
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



def processar_e_salvar_resultados(df_resultados, nome_arquivo_saida, alpha=0.05):
    df_resultados['Shapiro_Normal'] = df_resultados['Shapiro_P_Value'] > alpha
    df_resultados['K2_Normal'] = df_resultados['K2_P_Value'] > alpha
    df_resultados['KS_Normal'] = df_resultados['KS_P_Value'] > alpha

    df_resultados.to_csv(nome_arquivo_saida, index=False)

    print(f"\n💾 Resultados salvos em: {nome_arquivo_saida}")
    return df_resultados


# =========================================================
# Gráficos de distribuição do RMSE
# =========================================================
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
                df['nome_modelo'] == modelo,
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



# =========================================================
# Teste de múltiplas comparações (Tukey) + salvamento
# =========================================================
def executar_testes_multicomp_rmse(df, arquivo_figura):
    from statsmodels.stats.multicomp import MultiComparison

    df_long = df[['nome_modelo', 'mean_test_rmse']].dropna()

    mc = MultiComparison(df_long['mean_test_rmse'], df_long['nome_modelo'])
    tukey = mc.tukeyhsd()

    print("\n📊 Resultado Tukey HSD")
    print(tukey)

    fig = tukey.plot_simultaneous(figsize=(10, 6))
    plt.title("Tukey HSD – Comparação de RMSE entre Modelos")
    plt.savefig(f'fig/{arquivo_figura}', dpi=300, bbox_inches="tight")
    plt.close()

    print(f"🖼️ Gráfico Tukey salvo em: fig/{arquivo_figura}")



# =========================================================
# Execução principal
# =========================================================
if __name__ == "__main__":
    configurar_pandas()

    ARQUIVO_ENTRADA = "rmse.csv"
    ARQUIVO_RESULTADOS = "resultados_normalidade_rmse.csv"
    ARQUIVO_PDF = "distribuicao_rmse.pdf"
    ARQUIVO_IMG = "distribuicao_rmse.png"
    ARQUIVO_TUKEY = "tukey_rmse.png"

    MODELOS = [
        'regressao_rede_neural',
        'regressao_arvore_de_decisao',
        'regressao_random_florest',
        'regressao_s_v_r',
        'regressao_linear_lasso',
        'regressao_linear',
        'regressao_linear_ridge',
        'regressao_elastic_net',

    ]

    LABELS = [
        'Rede Neural',
        'Árvore de Decisão',
        'Random Forest',
        'SVR',
        'Lasso',
        'Linear',
        'Ridge',
        'Elastic Net',

    ]

    df_principal = carregar_dados(ARQUIVO_ENTRADA)

    df_testes = executar_testes_normalidade(df_principal, MODELOS)
    processar_e_salvar_resultados(df_testes, ARQUIVO_RESULTADOS)

    gerar_graficos_distribuicao_rmse(
        df_principal,
        MODELOS,
        LABELS,
        ARQUIVO_PDF,
        ARQUIVO_IMG
    )

    executar_testes_multicomp_rmse(
        df_principal,
        ARQUIVO_TUKEY
    )
