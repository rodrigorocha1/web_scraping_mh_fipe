from src_machine_learning.avaliador.avaliador import Avaliador
from src_machine_learning.estrategia_modelo.estrategia_modelo import EstrategiaModelo
from src_machine_learning.processador.processador import Processador, ModeloMachineLearning


class ProcessadorXGBoost(Processador):
    def __init__(self, estratregia_modelo: EstrategiaModelo, avaliador: Avaliador):
        super().__init__(estratregia_modelo, avaliador)

    def executar(self, opcao: int) -> ModeloMachineLearning:
        pass

    def _preparar_modelo(self, **kwargs) -> ModeloMachineLearning:
        pass
