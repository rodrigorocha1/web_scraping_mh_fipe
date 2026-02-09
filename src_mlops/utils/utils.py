import json
import os
from typing import Dict, Any, List

import numpy as np
import yaml


def carregar_dados_yaml_lista(parametro_modelo: str) -> List[Dict[str, Any]]:
    """
    Carrega os parâmetros do YAML e retorna como lista de dicionários

    :param parametro_modelo: chave dentro de 'model' do YAML (ex: 'parametros_treinamento_simples')
    :return: lista de dicionários com modelos e parâmetros
    """
    caminho_arquivo = os.path.join(os.getcwd(), "src_machine_learning", "config", "config.yaml")
    with open(caminho_arquivo, "r", encoding="utf-8") as file:
        config: Dict[str, Any] = yaml.safe_load(file)

    modelos_dict = config['model'].get(parametro_modelo, {})

    # Transforma em lista de dicionários sem usar for
    modelos_lista = list(map(lambda item: {"modelo": item[0], "parametros": item[1]}, modelos_dict.items()))

    return modelos_lista

def converter_numpy_para_list(obj):
    """
    Converte recursivamente ndarrays do NumPy para listas,
    para que sejam serializáveis em JSON.
    """
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    elif isinstance(obj, dict):
        return {k: converter_numpy_para_list(v) for k, v in obj.items()}
    elif isinstance(obj, list):
        return [converter_numpy_para_list(x) for x in obj]
    else:
        return obj


def salvar_json(dados: Dict[str, Any], diretorio: str, nome_arquivo: str, identacao: int = None):
    caminho_completo = os.path.join(os.getcwd(), diretorio, f'{nome_arquivo}.jsonl')
    dados_serializaveis = converter_numpy_para_list(dados)
    with open(caminho_completo, 'a', encoding='utf-8') as json_file:
        json_file.write(json.dumps(dados, ensure_ascii=False, indent=identacao) + '\n')