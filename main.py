import logging
import sys
import time
from typing import List, Tuple

from tqdm import tqdm

from src_machine_learning.avaliador.avaliador import Avaliador
from src_machine_learning.avaliador.avaliador_avore_decisao import AvaliadorArvoreDecisao
from src_machine_learning.avaliador.avaliador_floresta_aleatoria import AvaliadorFlorestaAleatoria
from src_machine_learning.avaliador.avaliador_rede_neural import AvaliadorRedeNeural
from src_machine_learning.avaliador.avaliador_regressao_linear import AvaliadorRegressaoLinear
from src_machine_learning.avaliador.avaliador_regressao_linear_regularizada import AvaliadorRegressaoLinearRegularizada
from src_machine_learning.avaliador.avaliador_svr import AvaliadorSVR
from src_machine_learning.estrategia_modelo.estrategia_modelo import EstrategiaModelo
from src_machine_learning.estrategia_modelo.estrategia_regressao_arvore_decisao import \
    EstrategiaRegressaoArvoreDeDecisao
from src_machine_learning.estrategia_modelo.estrategia_regressao_linear import EstrategiaRegressaoLinear
from src_machine_learning.estrategia_modelo.estrategia_regressao_linear_elastic_net import EstrategiaRegressaoElasticNet
from src_machine_learning.estrategia_modelo.estrategia_regressao_linear_lasso import EstrategiaRegressaoLinearLasso
from src_machine_learning.estrategia_modelo.estrategia_regressao_linear_ridge import EstrategiaRegressaoLinearRidge
from src_machine_learning.estrategia_modelo.estrategia_regressao_random_florest import EstrategiaRegressaoRandomFlorest
from src_machine_learning.estrategia_modelo.estrategia_regressao_rede_neural import EstrategiaRegressaoRedeNeural
from src_machine_learning.estrategia_modelo.estrategia_regressao_svr import EstrategiaRegressaoSVR
from src_machine_learning.processador.prepocessador_sklearn import PrepocessadorSklearnn
opcao = 2
opcao_execucao = 2
inicio_modelo = time.time()
modelos: List[Tuple[Avaliador, EstrategiaModelo]] = [
    # (AvaliadorArvoreDecisao(), EstrategiaRegressaoArvoreDeDecisao(opcao = opcao)),
    # (AvaliadorSVR(), EstrategiaRegressaoSVR(opcao = opcao)),
    # (AvaliadorRedeNeural(), EstrategiaRegressaoRedeNeural(opcao = opcao)),
    # (AvaliadorFlorestaAleatoria(), EstrategiaRegressaoRandomFlorest(opcao = opcao)),
    (AvaliadorRegressaoLinear(), EstrategiaRegressaoLinear(opcao=opcao)),
    (AvaliadorRegressaoLinearRegularizada(), EstrategiaRegressaoLinearLasso(opcao = opcao)),
    (AvaliadorRegressaoLinearRegularizada(), EstrategiaRegressaoLinearRidge(opcao = opcao)),
    (AvaliadorRegressaoLinearRegularizada(), EstrategiaRegressaoElasticNet(opcao = opcao)),
    (AvaliadorRegressaoLinear(), EstrategiaRegressaoLinear(opcao = opcao, polinomial=True)),
    (AvaliadorRegressaoLinearRegularizada(), EstrategiaRegressaoLinearLasso(polinomial=True)),
    (AvaliadorRegressaoLinearRegularizada(), EstrategiaRegressaoLinearRidge(opcao = opcao,polinomial=True)),
    (AvaliadorRegressaoLinearRegularizada(), EstrategiaRegressaoElasticNet(opcao = opcao, polinomial=True))
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
tempo_fim = time.time()
tempo_execucao_total = tempo_fim - inicio_modelo

fim_total = time.time()
tempo_total = fim_total - inicio_total
minutos_total = int(tempo_total // 60)
segundos_total = int(tempo_total % 60)
logging.info(f'Tempo de execução total : {minutos_total}:{segundos_total:02d} minutos')
