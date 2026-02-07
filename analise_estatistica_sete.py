import pandas as pd
import matplotlib.pyplot as plt
import json

# Load the data
file_path = 'dados/resultados_validacao_cruzada/regressao_arvore_de_decisao/resultado_validacao_cruzada_regressao_arvore_de_decisao.jsonl'
with open(file_path, 'r') as f:
    data = json.load(f)

# Extract feature importances
feature_importances = data.get('feature_importances', {})
df_importance = pd.DataFrame(list(feature_importances.items()), columns=['Feature', 'Importance'])
df_importance = df_importance.sort_values(by='Importance', ascending=True)

# Plotting top features (if there are too many, take the top 15)
top_features = df_importance.tail(15)

plt.figure(figsize=(10, 8))
plt.barh(top_features['Feature'], top_features['Importance'], color='skyblue')
plt.xlabel('Importância')
plt.ylabel('Atributo (Feature)')
plt.title('Importância das Variáveis no Modelo de Árvore de Decisão')
plt.tight_layout()
plt.savefig('importancia_variaveis.png')

print("Chart saved as importancia_variaveis.png")