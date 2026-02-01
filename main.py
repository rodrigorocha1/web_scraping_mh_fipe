import logging
import sys

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
    p.executar(2)
    break

