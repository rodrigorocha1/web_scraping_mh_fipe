from sklearn.datasets import load_diabetes
from sklearn.model_selection import GridSearchCV, train_test_split
from sklearn.tree import DecisionTreeRegressor
from sklearn.metrics import mean_squared_error

# 1️⃣ Carregar dataset
X, y = load_diabetes(return_X_y=True)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 2️⃣ Criar modelo
dtr = DecisionTreeRegressor(random_state=42)

# 3️⃣ Definir grid de parâmetros
param_grid = {
    'max_depth': [2, 4, 6, 8, None],
    'min_samples_split': [2, 5, 10]
}

# 4️⃣ Configurar GridSearchCV
grid_search_dtr = GridSearchCV(
    estimator=dtr,
    param_grid=param_grid,
    scoring='neg_root_mean_squared_error',  # RMSE negativo (o sklearn maximiza a métrica)
    cv=5,
    n_jobs=-1,
    return_train_score=True  # opcional, para ver scores de treino
)

# 5️⃣ Rodar busca
grid_search_dtr.fit(X_train, y_train)

# 6️⃣ Melhor modelo
dtr_otimizado = grid_search_dtr.best_estimator_
print(dtr_otimizado, type(dtr_otimizado))

# 7️⃣ Resultados
print("Melhor score (neg RMSE):", grid_search_dtr.best_score_)
print("Melhores parâmetros:", grid_search_dtr.best_params_)

# 8️⃣ DataFrame com todos os resultados
import pandas as pd
results_df = pd.DataFrame(grid_search_dtr.cv_results_)
print(results_df[['params', 'mean_test_score', 'mean_train_score', 'rank_test_score']])
