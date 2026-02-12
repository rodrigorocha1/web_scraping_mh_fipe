import logging
import sys
import time
from typing import List, Tuple

from tqdm import tqdm

from src_mlops.avaliador_mlops.avaliador_avore_decisao import AvaliadorArvoreDecisao
from src_mlops.avaliador_mlops.avaliador_floresta_aleatoria import AvaliadorFlorestaAleatoria
from src_mlops.avaliador_mlops.avaliador_rede_neural import AvaliadorRedeNeural
from src_mlops.avaliador_mlops.avaliador_regressao_linear import AvaliadorRegressaoLinear
from src_mlops.avaliador_mlops.avaliador_regressao_linear_regularizada import AvaliadorRegressaoLinearRegularizada
from src_mlops.avaliador_mlops.avaliador_svr import AvaliadorSVR
from src_mlops.estrategia_modelo.estrategia_regressao_arvore_decisao import \
    EstrategiaRegressaoArvoreDeDecisao
from src_mlops.estrategia_modelo.estrategia_regressao_linear import EstrategiaRegressaoLinear
from src_mlops.estrategia_modelo.estrategia_regressao_linear_elastic_net import EstrategiaRegressaoElasticNet
from src_mlops.estrategia_modelo.estrategia_regressao_linear_lasso import EstrategiaRegressaoLinearLasso
from src_mlops.estrategia_modelo.estrategia_regressao_linear_ridge import EstrategiaRegressaoLinearRidge
from src_mlops.estrategia_modelo.estrategia_regressao_random_florest import EstrategiaRegressaoRandomFlorest
from src_mlops.estrategia_modelo.estrategia_regressao_rede_neural import EstrategiaRegressaoRedeNeural
from src_mlops.estrategia_modelo.estrategia_regressao_svr import EstrategiaRegressaoSVR
from src_mlops.processador.prepocessador_sklearn_ml_ops import PrepocessadorSklearnn

opcao = 1
opcao_execucao = 3
inicio_modelo = time.time()
modelos: List[Tuple] = [
    (AvaliadorArvoreDecisao(), EstrategiaRegressaoArvoreDeDecisao(opcao=opcao)),
    (AvaliadorSVR(), EstrategiaRegressaoSVR(opcao=opcao)),
    (AvaliadorRedeNeural(), EstrategiaRegressaoRedeNeural(opcao=opcao)),
    (AvaliadorFlorestaAleatoria(), EstrategiaRegressaoRandomFlorest(opcao=opcao)),
    (AvaliadorRegressaoLinear(), EstrategiaRegressaoLinear(opcao=opcao)),
    (AvaliadorRegressaoLinearRegularizada(), EstrategiaRegressaoLinearLasso(opcao=opcao)),
    (AvaliadorRegressaoLinearRegularizada(), EstrategiaRegressaoLinearRidge(opcao=opcao)),
    (AvaliadorRegressaoLinearRegularizada(), EstrategiaRegressaoElasticNet(opcao=opcao)),

]

inicio_total = time.time()  # tempo total do script

for modelo in tqdm(
        modelos,
        desc=f"🔎 Treinando modelo  ",
        unit="modelo",
        file=sys.stdout,  # 👈 força exibição no terminal
        ncols=100  # 👈 largura fixa (opcional)
):
    avaliador, modelo_ml = modelo
    logging.info(f'Treinando modelo {modelo_ml.__class__.__name__.split(".")[-1]}')
    inicio_modelo = time.time()
    p = PrepocessadorSklearnn(
        avaliador=avaliador,
        estratregia_modelo=modelo_ml
    )
    p.executar(opcao_execucao)
    fim_modelo = time.time()

    tempo_modelo = fim_modelo - inicio_modelo
    minutos_modelo = int(tempo_modelo // 60)
    segundos_modelo = int(tempo_modelo % 60)
    logging.info(
        f'Tempo de execução do modelo {modelo_ml.__class__.__name__}: {minutos_modelo}:{segundos_modelo:02d} minutos'
    )
    break
tempo_fim = time.time()
tempo_execucao_total = tempo_fim - inicio_modelo

fim_total = time.time()
tempo_total = fim_total - inicio_total
minutos_total = int(tempo_total // 60)
segundos_total = int(tempo_total % 60)
logging.info(f'Tempo de execução total : {minutos_total}:{segundos_total:02d} minutos')
