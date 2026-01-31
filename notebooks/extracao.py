import logging
import sys
from time import sleep

from selenium import webdriver
from selenium.webdriver.chrome.service import Service
from selenium.webdriver.common.by import By
from selenium.webdriver.common.keys import Keys
from tqdm import tqdm
from webdriver_manager.chrome import ChromeDriverManager

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

formatter = logging.Formatter(
    "%(asctime)s | %(levelname)s | %(name)s | %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S"
)

# Handler para arquivo
file_handler = logging.FileHandler("selenium_fipe.log")
file_handler.setLevel(logging.INFO)
file_handler.setFormatter(formatter)

# Handler para terminal
console_handler = logging.StreamHandler(sys.stdout)
console_handler.setLevel(logging.INFO)
console_handler.setFormatter(formatter)

# Evita duplicação de logs
if not logger.handlers:
    logger.addHandler(file_handler)
    logger.addHandler(console_handler)
modelos_pesquisa = [
    "Peugeot",
    'Honda',
    "Mitsubishi",
    "Kia Motors",
    "Toyota"

]

for modelo_pesquisa in modelos_pesquisa:

    def abrir_nagegador():
        servico = Service(ChromeDriverManager().install())
        driver = webdriver.Chrome(service=servico)
        driver.maximize_window()
        driver.get("https://veiculos.fipe.org.br/")
        return driver


    driver = abrir_nagegador()
    sleep(5)


    def direcionar_barra_pesquisa(driver):
        element = driver.find_element(By.XPATH, "//*[@id='front']/div[1]/div[2]/ul/li[1]/a/div[2]")
        driver.execute_script("arguments[0].click();", element)


    direcionar_barra_pesquisa(driver)
    sleep(20)


    def digitar_marca(driver, marca_pesquisa_carro):
        driver.find_element(By.CLASS_NAME, "chosen-default").click()
        marca = driver.find_element(By.XPATH, '//*[@id="selectMarcacarro_chosen"]/div/div/input')
        marca.send_keys(marca_pesquisa_carro)
        marca.send_keys(Keys.ENTER)


    digitar_marca(driver, modelo_pesquisa)


    def pesquisar_ano_modelo(driver):
        elemento_web_modelo = driver.find_element(By.ID, "selectAnoModelocarro_chosen")
        elemento_web_modelo.click()
        lista_modelos = elemento_web_modelo.find_elements(By.TAG_NAME, "li")
        lista_modelos = [modelo.text for modelo in lista_modelos]
        elemento_web_modelo.click()
        return lista_modelos


    lista_modelos = pesquisar_ano_modelo(driver)
    sleep(5)


    def buscar_ano_modelo(driver, modelo):
        try:
            logger.info(f'Buscando para o modelo: {modelo}')
            driver.find_element(By.XPATH, '//*[@id="selectAnoModelocarro_chosen"]/a/span').click()

            caixa_texto_modelo_veiculo = driver.find_element(By.XPATH,
                                                             '//*[@id="selectAnoModelocarro_chosen"]/div/div/input')

            caixa_texto_modelo_veiculo.send_keys(modelo)
            caixa_texto_modelo_veiculo.send_keys(Keys.ENTER)
            elemento_web_ano_modelo = driver.find_element(By.ID, "selectAnocarro_chosen")
            elemento_web_ano_modelo.click()
            lista_ano_modelo = elemento_web_ano_modelo.find_elements(By.TAG_NAME, "li")
            logger.info(f'tamanho  da  lista: {len(lista_ano_modelo)}')
            resultado = []
            for modelo in lista_ano_modelo:
                sleep(1)
                resultado.append(modelo.text)

            elemento_web_ano_modelo.click()

            return resultado

        except Exception as e:
            print(f'Erro ao buscar ano modelo: {modelo}')
            logger.exception(f'Erro ao buscar ano modelo: {modelo}')
            caixa_texto_modelo_veiculo.clear()

            return []


    import pickle


    def gravar_pickle(objeto, caminho: str):
        """
        Grava qualquer objeto Python em um arquivo pickle (.pkl)
        """

        with open(caminho, 'wb') as arquivo:
            pickle.dump(objeto, arquivo)

        print(f'✅ Pickle gravado com sucesso em: {caminho}')


    logger.info(f'Tamanho da Lista de modelos: {len(lista_modelos)}')
    chave_pesquisa = [
        {
            'nome_marca': modelo_pesquisa,
            'nome_modelo': [
                {
                    'nome': modelo_carro_salvar,
                    'flag_pesquisado': False,
                    'ano_modelo': [
                        {
                            'nome_ano_modelo': buscar_ano_modelo(driver, modelo_carro_salvar),
                            'flag_pesquisado': False
                        }
                    ]
                }
                for modelo_carro_salvar in tqdm(
                    lista_modelos,
                    desc=f"🔎 Buscando modelos {modelo_pesquisa}",
                    unit="modelo",
                    file=sys.stdout,  # 👈 força exibição no terminal
                    ncols=100  # 👈 largura fixa (opcional)
                )
            ],
        }
    ]
    import json

    # GM - Chevrolet

    print(json.dumps(chave_pesquisa, indent=4, ensure_ascii=False))
    gravar_pickle(objeto=chave_pesquisa, caminho=f'{modelo_pesquisa.lower().replace(" ", "_").replace("-", "_")}.pkl')
    driver.quit()
    sleep(10)
