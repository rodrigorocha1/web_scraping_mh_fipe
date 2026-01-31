from abc import ABC, abstractmethod


class EstrategiaModelo(ABC):

    @abstractmethod
    def realizar_estrategia(self) -> str:
        pass