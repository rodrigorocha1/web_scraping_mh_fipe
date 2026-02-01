import logging
import sys
import time
from typing import List, Tuple

from tqdm import tqdm

from src_machine_learning.avaliador.avaliador import Avaliador
from src_machine_learning.avaliador.avaliador_regressao_linear_regularizada import AvaliadorRegressaoLinearRegularizada
from src_machine_learning.estrategia_modelo.estrategia_modelo import EstrategiaModelo
from src_machine_learning.estrategia_modelo.estrategia_regressao_linear_elastic_net import EstrategiaRegressaoElasticNet
from src_machine_learning.processador.prepocessador_sklearn import PrepocessadorSklearnn

inicio_modelo = time.time()
modelos: List[Tuple[Avaliador, EstrategiaModelo]] = [
    # (AvaliadorArvoreDecisao(), EstrategiaRegressaoArvoreDeDecisao()),
    # (AvaliadorSVR(), EstrategiaRegressaoSVR()),
    # (AvaliadorRedeNeural(), EstrategiaRegressaoRedeNeural()),
    # (AvaliadorFlorestaAleatoria(), EstrategiaRegressaoRandomFlorest()),
    # (AvaliadorRegressaoLinear(), EstrategiaRegressaoLinear()),
    # (AvaliadorRegressaoLinearRegularizada(), EstrategiaRegressaoLinearLasso())
    # (AvaliadorRegressaoLinearRegularizada(), EstrategiaRegressaoLinearRidge())
    (AvaliadorRegressaoLinearRegularizada(), EstrategiaRegressaoElasticNet())
]
opcao = 1
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
    p.executar(opcao)
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
