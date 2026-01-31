from typing import Tuple, List, Dict, Any, Generator

import requests
from dotenv import load_dotenv

from src_api.context.contexto_api import ContextoApi
from src_api.corrente import Corrente




class CorrenteAno(Corrente):

    def __init__(self, vehicle_type: str, brand_id: str):
        self.__vehicle_type = vehicle_type
        self.__brand_id = brand_id
        super().__init__()

    def executar_processo(self, contexto: ContextoApi) -> bool:
        url = f"https://fipe.parallelum.com.br/api/v2/{self.__vehicle_type}/brands/{self.__brand_id}/models/"

        req = requests.get(url=url, headers={"Authorization": f"Bearer {self._keys}",
                                             "Content-Type": "application/json"})
        resultado = req.json()
        contexto.lista_modelos = resultado

        return True
