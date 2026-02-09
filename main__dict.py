lista_resultados = [
    {'mae': 14259.9215424775, 'rmse': 28439.905745708817, 'medae': 5716.454545454544, 'smape': 15.285206133888044,
     'r2': 0.8901430944089287, 'bias': 635.417112828146, 'accuracy_erro_10_pct': 0.48716496791241976,
     'preco_medio_real': 89551.59777274443, 'preco_medio_previsto': 90187.01488557257, 'n_amostras': 5298,
     'profundidade_arvore': 20, 'numero_folhas': 1784, 'numero_nos': 3567,
     'feature_importances': {'ano_modelo': 0.4964001734489032, 'tipo_combustivel_Flex': 0.2166522288456771,
                             'motor_cilindrada': 0.163210870800901, 'marca_Mercedes-Benz': 0.02658383690016241,
                             'marca_Audi': 0.01984145459970742, 'tipo_combustivel_Gasolina': 0.009166497812098991,
                             'tipo_transmissao_Automático': 0.00649778685647278, 'marca_Volvo': 0.006064540826547765,
                             'tipo_combustivel_Elétrico': 0.005645876452204685,
                             'tipo_combustivel_Diesel': 0.005524454885674874,
                             'tipo_transmissao_Manual': 0.005448237861964435, 'turbo_Não': 0.0053678838820224685,
                             'turbo_Sim': 0.004310953159175607, 'marca_Toyota': 0.003985082519719889,
                             'marca_Ford': 0.0032225834974610127, 'marca_VW - VolksWagen': 0.0029143645539586975,
                             'marca_Jeep': 0.002729004029611892, 'marca_GM - Chevrolet': 0.002532551060886248,
                             'tipo_combustivel_Híbrido': 0.002330741968308142, 'marca_IVECO': 0.0020388070014770066,
                             'marca_Nissan': 0.0014732956126245636, 'marca_Honda': 0.0014187486393755831,
                             'marca_JAC': 0.0012763315446690817, 'marca_Kia Motors': 0.001032898003417072,
                             'marca_Fiat': 0.0009131626018352089, 'marca_Mitsubishi': 0.0008749237941968427,
                             'marca_Renault': 0.0007503474725804406, 'marca_Troller': 0.00047192185469272885,
                             'marca_BYD': 0.00040397327119367116, 'marca_Hyundai': 0.00037810218972813035,
                             'marca_Citroën': 0.0002738559104265185, 'marca_Peugeot': 0.00026026550402192357,
                             'tipo_combustivel_infrequent_sklearn': 2.753985736126891e-06,
                             'tipo_combustivel_Álcool': 1.488652566585525e-06, 'modelo_infrequent_sklearn': 0.0}},
    {
        "mae": 29449.71865813922,
        "rmse": 43109.36215588038,
        "medae": 20845.79593985407,
        "smape": 50.38603590075203,
        "r2": 0.7475855284146133,
        "bias": 1448.3425757302336,
        "accuracy_erro_10_pct": 0.17818044545111364,
        "preco_medio_real": 89551.59777274443,
        "preco_medio_previsto": 90999.94034847466,
        "n_amostras": 5298,
        "modelo": "ElasticNet",
        "alpha": 0.001,
        "intercepto": 118725.78800012296,
        "coeficientes": {
            "motor_cilindrada": 25415.220965921446,
            "ano_modelo": 60316.679167653965,
            "marca_Audi": 39763.488240223036,
            "marca_BYD": -36834.53977082254,
            "marca_Citroën": -16783.149377585494,
            "marca_Fiat": -4788.699903216773,
            "marca_Ford": -305.4911954680729,
            "marca_GM - Chevrolet": 633.4257106147538,
            "marca_Honda": 5802.775568579201,
            "marca_Hyundai": -12040.851328238396,
            "marca_IVECO": 49615.15476193013,
            "marca_JAC": -39246.688003157185,
            "marca_Jeep": 10052.879199280284,
            "marca_Kia Motors": -20392.400611845263,
            "marca_Mercedes-Benz": 54321.7645134942,
            "marca_Mitsubishi": -8097.917393809922,
            "marca_Nissan": -14528.588075986487,
            "marca_Peugeot": -11534.02852045428,
            "marca_Renault": -6836.663471936395,
            "marca_Toyota": 5978.414885015962,
            "marca_Troller": 6132.0288860176715,
            "marca_VW - VolksWagen": -2219.235158072587,
            "marca_Volvo": 1311.3200333687391,
            "modelo_infrequent_sklearn": 0.0,
            "tipo_combustivel_Diesel": -18109.657017527723,
            "tipo_combustivel_Elétrico": 66789.7485052727,
            "tipo_combustivel_Flex": -61673.54760568535,
            "tipo_combustivel_Gasolina": -18588.958596637334,
            "tipo_combustivel_Híbrido": 55967.14391883963,
            "tipo_combustivel_Álcool": 2570.6577348532533,
            "tipo_combustivel_infrequent_sklearn": -26954.380009085948,
            "tipo_transmissao_Automático": -3983.4509783509925,
            "tipo_transmissao_Manual": 3981.429376985895,
            "turbo_Não": -5495.467313896471,
            "turbo_Sim": 5494.648282199539
        },
        "l1_ratio": 0.5,
        "data_coleta": "02/02/2026 08:38:55",
        "nome_modelo": "regressao_elastic_net"
    }
]
# Lista de chaves que queremos checar

for resultado in lista_resultados:
    for dado in resultado.items():

        if isinstance(dado[1], dict):
            print('verdadeiro')
            for item in dado[1].items():
                print(item)
        else:
            print('falso')
            print(dado)

