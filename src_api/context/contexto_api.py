import os

import requests
from typing import Dict, Any


def buscar_preco_fipe(
    tipo_veiculo: str,
    marca_id: str,
    modelo_id: str,
    ano_codigo: str,
    token: str | None = None
) -> Dict[str, Any]:
    """
    Consulta o preço de um veículo na API FIPE.

    :param tipo_veiculo: carros | motos | caminhoes
    :param marca_id: ID da marca (ex: '59')
    :param modelo_id: ID do modelo (ex: '5940')
    :param ano_codigo: Código do ano (ex: '2014-3')
    :param token: Token de assinatura (opcional)
    :return: JSON da resposta da API
    """

    url = (
        f"https://parallelum.com.br/fipe/api/v1/"
        f"{tipo_veiculo}/marcas/{marca_id}/modelos/{modelo_id}/anos/{ano_codigo}"
    )

    headers = {}
    if token:
        headers["X-Subscription-Token"] = os.environ['TOKEN']

    response = requests.get(url, headers=headers, timeout=10)

    # response.raise_for_status()  # levanta exceção se HTTP != 200

    return response.json()




if __name__ == "__main__":
    dados = buscar_preco_fipe(
        tipo_veiculo="carros",
        marca_id="59",
        modelo_id="5940",
        ano_codigo="2014-3"
    )

    print(dados)