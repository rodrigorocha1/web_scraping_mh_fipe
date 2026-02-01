from sklearn.datasets import load_diabetes
from sklearn.model_selection import cross_val_score, cross_validate
from sklearn.linear_model import LinearRegression

# Carregar dataset de regressão
X, y = load_diabetes(return_X_y=True)
print(type(X), type(y))
# Criar modelo
model = LinearRegression()
print(X)
print(y)
# -----------------------------
# Usando cross_val_score
# -----------------------------
scores = cross_val_score(model, X, y, cv=5, scoring='r2')  # Métrica R²
print(scores)
print("cross_val_score:")
print("R² por fold:", scores)
print("Média R²:", scores.mean())
print()

# -----------------------------
# Usando cross_validate
# -----------------------------
scoring = ['r2', 'neg_mean_squared_error']  # Podemos usar múltiplas métricas
results = cross_validate(model, X, y, cv=5, scoring=scoring, return_train_score=True)

print("cross_validate:")
print("R² nos testes:", results['test_r2'])
print("MSE nos testes:", -results['test_neg_mean_squared_error'])  # inverter sinal
print("R² nos treinamentos:", results['train_r2'])
print("Tempo de treino por fold:", results['fit_time'])
