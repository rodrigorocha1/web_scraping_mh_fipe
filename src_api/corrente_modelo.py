import json
import os
from time import sleep
from typing import List, Dict, Any

import requests
from dotenv import load_dotenv

load_dotenv()


class FipeAPI:

    def __init__(self):
        self.__keys = os.getenv('TOKEN')

    def executar_busca_modelos(self, brand_id: str, ) -> List[Dict[str, Any]]:
        url = f"https://fipe.parallelum.com.br/api/v2/cars/brands/{brand_id}/models"

        req = requests.request("GET",
            url=url,
            headers={
                "X-Subscription-Token": f"{self.__keys}",
                "Content-Type": "application/json"}
        )
        resultado = req.json()
        print(resultado)


        return resultado

    def executar_busca_ano(self, brand_id: str, model_id: str) -> List[Dict[str, Any]]:
        url = f"https://fipe.parallelum.com.br/api/v2/cars/brands/{brand_id}/models/{model_id}/years"
        req = requests.request("GET",
            url=url,
            headers={
                "X-Subscription-Token": f"{self.__keys}",
                "Content-Type": "application/json"}
        )
        resultado = req.json()


        return resultado

    def buscar_valor(self, brand_id: str, model_id: str, year_id: str) -> Dict[str, Any]:
        url = f"https://parallelum.com.br/fipe/api/v1/carros/marcas/{brand_id}/modelos/{model_id}/anos/{year_id}"
        sleep(2)
        req = requests.request("GET",
            url=url,

            headers={
                "X-Subscription-Token": f"{os.environ['TOKEN']}",
                "Content-Type": "application/json"},
        )
        resultado = req.json()

        print('Dentro de buscar_valor')

        return resultado

    @staticmethod
    def salvar_json(resultado: Dict[str, Any], diretorio: str, nome_arquivo: str):
        caminho_completo = os.path.join(os.getcwd(), diretorio, f'{nome_arquivo}.jsonl')

        with open(caminho_completo, 'a', encoding='utf-8') as json_file:
            json_file.write(json.dumps(resultado, ensure_ascii=False) + '\n')
