from src_machine_learning.processador.processador import Processador


p = Processador()

dataframe = p.abrir_dataframe()
dataframe = p.fazer_processamento(dataframe)
dataframe = p.realizar_engenharia_atributos(dataframe)
# print(dataframe.head())
