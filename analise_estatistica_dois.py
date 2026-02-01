import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import norm
import scipy.stats as stats
from matplotlib.backends.backend_pdf import PdfPages
import numpy as np
import os


def configurar_pandas():
    """Configura as opções de exibição do Pandas."""
    pd.set_option('display.max_columns', None)
    pd.set_option('display.max_rows', None)
    pd.set_option('display.width', 1000)
    pd.set_option('display.max_colwidth', 50)
    pd.set_option('display.precision', 4)


def carregar_dados(caminho_arquivo, separador='|'):
    """Carrega o dataset e exibe informações básicas."""
    df = pd.read_csv(caminho_arquivo, sep=separador)
    print("Dados carregados. Primeiras linhas:")
    print(df.head())
    print("\nInformações do DataFrame:")
    print(df.info())

    # Check iterations
    contagem_iteracoes = df['iteracao'].value_counts()
    print(f"\nNúmero de iterações únicas: {len(contagem_iteracoes)}")
    return df


def executar_testes_normalidade(df, modelos):
    """
    Executa testes de normalidade (Shapiro, K2, KS) para cada iteração e modelo.
    """
    resultados = []
    iteracoes = sorted(df['iteracao'].unique())

    print(f"\nIniciando testes de normalidade para {len(iteracoes)} iterações...")

    for iteracao in iteracoes:
        df_iter = df[df['iteracao'] == iteracao]
        for modelo in modelos:
            dados = df_iter[modelo]

            # Shapiro-Wilk
            stat_sw, p_sw = stats.shapiro(dados)

            # D'Agostino's K^2 test
            stat_k2, p_k2 = stats.normaltest(dados)

            # Kolmogorov-Smirnov test
            media, desvio = dados.mean(), dados.std()
            stat_ks, p_ks = stats.kstest(dados, 'norm', args=(media, desvio))

            resultados.append({
                'Iteracao': iteracao,
                'Modelo': modelo,
                'Shapiro_Stat': stat_sw, 'Shapiro_P_Value': p_sw,
                'K2_Stat': stat_k2, 'K2_P_Value': p_k2,
                'KS_Stat': stat_ks, 'KS_P_Value': p_ks
            })

    return pd.DataFrame(resultados)


def processar_e_salvar_resultados(df_resultados, nome_arquivo_saida, alpha=0.05):
    """
    Adiciona colunas de interpretação e salva os resultados em CSV.
    """
    df_resultados['Shapiro_Normal'] = df_resultados['Shapiro_P_Value'] > alpha
    df_resultados['K2_Normal'] = df_resultados['K2_P_Value'] > alpha
    df_resultados['KS_Normal'] = df_resultados['KS_P_Value'] > alpha

    df_resultados.to_csv(nome_arquivo_saida, index=False)
    print(f"\nResultados dos testes salvos em: {nome_arquivo_saida}")
    print(df_resultados.head())
    return df_resultados


def gerar_graficos_distribuicao(df, modelos, labels_modelos, nome_pdf, nome_imagem_exemplo):
    """
    Gera gráficos de distribuição (Hist + KDE + Normal) para cada iteração em um PDF.
    """
    iteracoes = sorted(df['iteracao'].unique())
    print(f"\nGerando gráficos para {len(iteracoes)} iterações...")

    # Garante que o diretório existe se houver subpasta
    if '/' in nome_pdf:
        os.makedirs(os.path.dirname(nome_pdf), exist_ok=True)

    with PdfPages(nome_pdf) as pdf:
        for i, num_iter in enumerate(iteracoes):
            fig, axes = plt.subplots(2, 2, figsize=(12, 10))
            fig.suptitle(f'Distribuição Normal dos Resíduos - Iteração {num_iter}', fontsize=16)

            df_iter = df[df['iteracao'] == num_iter]

            for j, (modelo, label) in enumerate(zip(modelos, labels_modelos)):
                ax = axes.flatten()[j]
                dados = df_iter[modelo].dropna()

                # Histograma e KDE
                sns.histplot(dados, kde=True, stat="density", ax=ax, color='skyblue', label='Resíduos (KDE)')

                # Curva Normal Ajustada
                mu, std = norm.fit(dados)
                xmin, xmax = ax.get_xlim()
                x = np.linspace(xmin, xmax, 100)
                p = norm.pdf(x, mu, std)
                ax.plot(x, p, 'r', linewidth=2, label=f'Normal ($\mu$={mu:.0f}, $\sigma$={std:.0f})')

                ax.set_title(label)
                ax.legend(loc='best', fontsize='small')

            plt.tight_layout(rect=[0, 0.03, 1, 0.95])

            pdf.savefig(fig)

            # Salva a primeira iteração como exemplo
            if i == 0:
                fig.savefig(nome_imagem_exemplo)

            plt.close(fig)

    print(f"Arquivo PDF gerado: {nome_pdf}")
    print(f"Imagem de exemplo gerada: {nome_imagem_exemplo}")

def executar_testes_multicomp(df_principal):
    iteracao = df_principal['iteracao'].to_list()
    resultados_arvore = df_principal['residuos_totais_arvore_decisao'].to_list()
    resultados_random = df_principal['residuos_totais_random_florest'].to_list()
    resultados_svr = df_principal['residuos_totais_regressao_svr'].to_list()
    resultados_rede = df_principal['residuos_totais_rede_neural'].to_list()

    # Obter o número de iterações para garantir que a multiplicação da lista 'algoritmo' esteja correta
    n = len(df_principal)

    # 2. Montar o dicionário (Corrigindo o tamanho da lista 'iteracao')
    dados = {
        # Repetimos a lista de iterações 4 vezes para alinhar com os 4 algoritmos empilhados
        'iteracao': iteracao * 4,

        # Criamos a lista de nomes dinamicamente baseada no tamanho 'n'
        'algoritmo': ['arvore_decisao'] * n + ['regressao_svr'] * n + ['random_florest'] * n + ['rede_neural'] * n,

        # Concatenamos os resultados na mesma ordem dos nomes acima
        'resultados': resultados_arvore + resultados_svr + resultados_random + resultados_rede
    }
    df_final = pd.DataFrame(dados)

    print(df_final.head())
    print(df_final.info())

    from statsmodels.stats.multicomp import MultiComparison
    compara_algoritmos = MultiComparison(df_final['resultados'], df_final['algoritmo'])
    teste_estatistico = compara_algoritmos.tukeyhsd()
    print(teste_estatistico)
    teste_estatistico.plot_simultaneous()
    plt.show()


# --- Execução Principal ---
if __name__ == "__main__":
    configurar_pandas()

    ARQUIVO_ENTRADA = 'residuos.csv'
    ARQUIVO_RESULTADOS = 'resultados_normalidade_refatorado.csv'
    ARQUIVO_PDF = 'distribuicao_normal_refatorado.pdf'
    ARQUIVO_IMG = 'distribuicao_exemplo_refatorado.png'

    MODELOS = [
        'residuos_totais_arvore_decisao',
        'residuos_totais_random_florest',
        'residuos_totais_regressao_svr',
        'residuos_totais_rede_neural'
    ]
    LABELS = ['Árvore de Decisão', 'Random Forest', 'SVR', 'Rede Neural']

    # 1. Carregar
    df_principal = carregar_dados(ARQUIVO_ENTRADA)


    # # 2. Testar Normalidade
    # df_testes = executar_testes_normalidade(df_principal, MODELOS)
    #
    # # 3. Salvar Resultados
    # processar_e_salvar_resultados(df_testes, ARQUIVO_RESULTADOS)
    #
    # # 4. Gerar Gráficos
    # gerar_graficos_distribuicao(df_principal, MODELOS, LABELS, ARQUIVO_PDF, ARQUIVO_IMG)

    # 1. Extrair as listas
    executar_testes_multicomp(df_principal)

    # 3. Converter para DataFrame
