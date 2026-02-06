import re

import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split

# =========================================================
# CONFIGURAÇÕES
# =========================================================

CAMINHO_CSV = "tabela-fipe-329.csv"
TARGET = "preco"

MARCAS_IDS = [
    6, 238, 23, 13, 21, 22, 25, 26, 208, 177,
    29, 31, 39, 41, 43, 44, 48, 56, 59, 57, 58
]

COLUNAS_RENAME = {
    "Type": "tipo",
    "Brand_Code": "codigo_marca",
    "Brand_Value": "marca",
    "Model_Code": "codigo_modelo",
    "Model_value": "modelo",
    "Year_Code": "codigo_ano",
    "Year_Value": "ano_modelo",  # vem como "2020 Flex"
    "Fipe_Code": "codigo_fipe",
    "Fuel_Letter": "sigla_combustivel",
    "Fuel_Type": "tipo_combustivel",
    "Price": "preco"
}

COLUNAS_CATEGORICAS = [
    "marca",
    "modelo",
    "tipo_combustivel",
    "tipo_transmissao",
    "turbo"
]

FEATURES_NUMERICAS = [
    "motor_cilindrada",
    "ano_modelo"
]

FEATURES = FEATURES_NUMERICAS + COLUNAS_CATEGORICAS


# =========================================================
# PRÉ-PROCESSAMENTO BÁSICO
# =========================================================

def filtrar_e_renomear(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df = df[df["Brand_Code"].isin(MARCAS_IDS)]
    df = df.rename(columns=COLUNAS_RENAME)
    return df


# =========================================================
# FUNÇÕES DE EXTRAÇÃO / LIMPEZA
# =========================================================

def extrair_ano(valor):
    """
    Extrai apenas o ano numérico de strings como:
    '2020 Flex', '2019 Gasolina', '2022 Diesel'
    """
    if pd.isna(valor):
        return None

    match = re.search(r"\b(19|20)\d{2}\b", str(valor))
    if match:
        return int(match.group())

    return None


def extrair_motor(modelo):
    if pd.isna(modelo):
        return None

    match = re.search(r"(\d\.\d)", str(modelo))
    return float(match.group()) if match else None


def extrair_transmissao(modelo):
    modelo = str(modelo).lower()

    automaticos = [
        "aut.", "automático", "automatico", "s-tronic",
        "tip.", "tiptronic", "dsg", "cvt", "powershift"
    ]

    return "Automático" if any(x in modelo for x in automaticos) else "Manual"


def extrair_turbo(modelo):
    modelo = str(modelo).lower()

    turbo_keywords = [
        "turbo", "tfsi", "tsi", "t-jet", "tb", "biturbo",
        "kompressor", "compressor", "thp", "tdi", "cdi",
        "cgi", "ecoboost", "tgdi", "t-gdi", "jtd", "hdi",
        "d-4d", "multijet", "tce", "crdi", "t200", "t270"
    ]

    if any(k in modelo for k in turbo_keywords):
        return "Sim"

    if re.search(r"\b[0-9]\.[0-9]\s?t\b", modelo):
        return "Sim"

    return "Não"


# =========================================================
# ENGENHARIA DE ATRIBUTOS (VERSÃO CORRIGIDA)
# =========================================================

def engenharia_atributos(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()

    # 🔹 Extrair atributos
    df["motor_cilindrada"] = df["modelo"].apply(extrair_motor)
    df["ano_modelo"] = df["ano_modelo"].apply(extrair_ano)
    df["tipo_transmissao"] = df["modelo"].apply(extrair_transmissao)
    df["turbo"] = df["modelo"].apply(extrair_turbo)

    # 🔹 Garantir tipos numéricos
    df["ano_modelo"] = pd.to_numeric(df["ano_modelo"], errors="coerce")
    df["motor_cilindrada"] = pd.to_numeric(df["motor_cilindrada"], errors="coerce")

    # 🔹 Converter categóricas
    for col in COLUNAS_CATEGORICAS:
        df[col] = df[col].astype("category")

    return df


def preparar_para_xgboost(df: pd.DataFrame):
    df = df.copy()
    feature_types = []

    for col in df.columns:
        if pd.api.types.is_categorical_dtype(df[col]):
            df[col] = df[col].cat.codes
            feature_types.append("c")
        else:
            feature_types.append("q")  # quantitativa

    return df, feature_types


# =========================================================
# PIPELINE PRINCIPAL
# =========================================================

def main():
    print("🔹 Carregando CSV...")
    df = pd.read_csv(CAMINHO_CSV)

    print("🔹 Filtrando marcas e renomeando colunas...")
    df = filtrar_e_renomear(df)

    print("🔹 Engenharia de atributos...")
    df = engenharia_atributos(df)

    # 🔹 Remover registros inválidos
    df = df.dropna(subset=FEATURES + [TARGET])

    X = df[FEATURES]
    y = df[TARGET]

    print("\n📊 Tipos finais das features:")
    print(X.dtypes)

    print("\n🔹 Split treino / validação...")
    X_train, X_valid, y_train, y_valid = train_test_split(
        X, y, test_size=0.2, random_state=42
    )
    for col in COLUNAS_CATEGORICAS:
        X_train[col] = X_train[col].astype("category")
        X_valid[col] = X_valid[col].astype("category")
    print(X_train.dtypes)
    print("🔹 Criando DMatrix...")
    X_train_xgb, feature_types = preparar_para_xgboost(X_train)
    X_valid_xgb, _ = preparar_para_xgboost(X_valid)

    dtrain = xgb.DMatrix(
        X_train_xgb,
        label=y_train,
        feature_types=feature_types
    )

    dvalid = xgb.DMatrix(
        X_valid_xgb,
        label=y_valid,
        feature_types=feature_types
    )
    params = {
        "objective": "reg:squarederror",
        "eval_metric": "rmse",
        "learning_rate": 0.05,
        "max_depth": 6,
        "subsample": 0.8,
        "colsample_bytree": 0.8,
        "tree_method": "hist",
        "seed": 42
    }

    print("🚀 Treinando XGBoost...")
    model = xgb.train(
        params=params,
        dtrain=dtrain,
        num_boost_round=2000,
        evals=[(dtrain, "train"), (dvalid, "valid")],
        early_stopping_rounds=50,
        verbose_eval=100
    )

    print("🔹 Avaliando modelo...")
    preds = model.predict(dvalid)
    rmse = np.sqrt(mean_squared_error(y_valid, preds))
    print(f"\n✅ RMSE validação: R$ {rmse:,.2f}")

    model.save_model("xgboost_fipe.json")
    print("\n💾 Modelo salvo em xgboost_fipe.json")

    # =====================================================
    # EXEMPLO DE PREDIÇÃO
    # =====================================================

    novo = pd.DataFrame([{
        "motor_cilindrada": 2.0,
        "ano_modelo": 2022,
        "marca": "VW - VolksWagen",
        "modelo": "Jetta 2.0 TSI",
        "tipo_combustivel": "Gasolina",
        "tipo_transmissao": "Automático",
        "turbo": "Sim"
    }])

    for col in COLUNAS_CATEGORICAS:
        novo[col] = novo[col].astype("category")

    dnovo = xgb.DMatrix(novo, enable_categorical=True)
    preco = model.predict(dnovo)[0]

    print(f"\n💰 Preço estimado: R$ {preco:,.2f}")


# =========================================================
# ENTRYPOINT
# =========================================================

if __name__ == "__main__":
    main()
