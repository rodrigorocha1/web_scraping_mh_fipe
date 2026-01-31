import os
from abc import ABC, abstractmethod
from typing import Optional, List, Dict, Any, Tuple
from dotenv import load_dotenv
from src_api.context.contexto_api import ContextoApi

load_dotenv()


class Corrente(ABC):

    def __init__(self) -> None:
        self._next_handler: Optional["Handler"] = None
        self._keys = os.getenv('TOKEN')

    def set_next(self, handler: "Handler"):
        self._next_handler = handler
        return handler

    def handler(self, context: ContextoApi) -> None:
        if self.executar_processo(context):

            if self._next_handler:
                self._next_handler.handler(context)
            else:
                pass
        else:
            pass

    @abstractmethod
    def executar_processo(self, contexto: ContextoApi) -> bool:
        pass
