import pandas as pd

from src_machine_learning.avaliador.avaliador_avore_decisao import AvaliadorArvoreDecisao
from src_machine_learning.estrategia_modelo.estrategia_regressao_arvore_decisao import \
    EstrategiaRegressaoArvoreDeDecisao
from src_machine_learning.processador.prepocessador_sklearn import PrepocessadorSklearnn

p = PrepocessadorSklearnn(
    avaliador=AvaliadorArvoreDecisao(),
    estratregia_modelo=EstrategiaRegressaoArvoreDeDecisao()
)
p.executar(1)

