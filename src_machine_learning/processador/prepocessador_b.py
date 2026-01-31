from src_machine_learning.processador.processador import Processador, ModeloMachineLearning
from sklearn.pipeline import Pipeline


class PrepocessadorB(Processador[str]):




    def __init__(self):
        super().__init__()

    def preparar_modelo(self, **kwargs) -> str:
        return 'A'

    def executar(self) -> ModeloMachineLearning:
        pass

