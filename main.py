import logging
import sys
import time

from tqdm import tqdm

from src_machine_learning.avaliador.avaliador_avore_decisao import AvaliadorArvoreDecisao
from src_machine_learning.avaliador.avaliador_floresta_aleatoria import AvaliadorFlorestaAleatoria
from src_machine_learning.avaliador.avaliador_rede_neural import AvaliadorRedeNeural
from src_machine_learning.avaliador.avaliador_svr import AvaliadorSVR
from src_machine_learning.estrategia_modelo.estrategia_regressao_arvore_decisao import \
    EstrategiaRegressaoArvoreDeDecisao
from src_machine_learning.estrategia_modelo.estrategia_regressao_random_florest import EstrategiaRegressaoRandomFlorest
from src_machine_learning.estrategia_modelo.estrategia_regressao_rede_neural import EstrategiaRegressaoRedeNeural
from src_machine_learning.estrategia_modelo.estrategia_regressao_svr import EstrategiaRegressaoSVR
from src_machine_learning.processador.prepocessador_sklearn import PrepocessadorSklearnn

inicio_modelo = time.time()
modelos = [
    (AvaliadorArvoreDecisao(), EstrategiaRegressaoArvoreDeDecisao()),
    (AvaliadorSVR(), EstrategiaRegressaoSVR()),
    (AvaliadorRedeNeural(), EstrategiaRegressaoRedeNeural()),
    (AvaliadorFlorestaAleatoria(), EstrategiaRegressaoRandomFlorest()),
]

for modelo in tqdm(
        modelos,
        desc=f"🔎 Treinando modelo  ",
        unit="modelo",
        file=sys.stdout,  # 👈 força exibição no terminal
        ncols=100  # 👈 largura fixa (opcional)
):
    avaliador, modelo_ml = modelo
    logging.info(f'Treinando modelo {modelo_ml.__class__.__name__.split(".")[-1]}')
    p = PrepocessadorSklearnn(
        avaliador=avaliador,
        estratregia_modelo=modelo_ml
    )
    p.executar(1)
    end_time = time.time()
    elapsed_time = end_time - inicio_modelo
    minutes = int(elapsed_time // 60)
    seconds = int(elapsed_time % 60)

    logging.info(
        f'Tempo de execução do modelo {modelo_ml.__class__.__name__.split(".")[-1]}: : {minutes}:{seconds:02d} segundos')
tempo_fim = time.time()
tempo_execucao_total = tempo_fim - inicio_modelo

minutos_total = int(tempo_execucao_total // 60)
segundos_total = int(tempo_execucao_total % 60)
logging.info(f'Tempo de execução total : {minutos_total}:{segundos_total:02d} segundos')
