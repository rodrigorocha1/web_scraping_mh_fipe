from abc import ABC, abstractmethod
from io import BytesIO
from typing import Dict, Any, List

import numpy as np
import sklearn
from pandas import DataFrame, Series
from sklearn.base import BaseEstimator
from sklearn.model_selection import GridSearchCV
from sklearn.inspection import permutation_importance
from io import BytesIO
from PIL import Image
import matplotlib.pyplot as plt
import numpy as np
import mlflow
from pandas import DataFrame, Series
from sklearn.pipeline import Pipeline


class Avaliador(ABC):

    @abstractmethod
    def obter_dados_curva_validacao(self, pipeline: BaseEstimator, X_train: DataFrame, y_train: Series) -> \
            Dict[str, Any]:
        pass

    @abstractmethod
    def obter_resultados_modelo(self, pipeline: sklearn.pipeline.Pipeline, y_test: Series,
                                y_pred: np.ndarray[Any, np.dtype[Any]]) -> \
            Dict[str, Any]:
        pass

    @abstractmethod
    def gerar_grafico_underfit_overfit(self, metricas: Dict[str, Any]) -> Dict[str, BytesIO]:
        pass

    @abstractmethod
    def obter_resultado_grid_search(self, grid_search: GridSearchCV) -> Dict[str, Any]:
        pass


    @staticmethod
    def __obter_nomes_features_p(preprocessor) -> List[str]:
        feature_names = []

        for nome, transformer, cols in preprocessor.transformers_:
            if nome == 'remainder' and transformer == 'drop':
                continue

            if hasattr(transformer, 'get_feature_names_out'):
                feature_names.extend(
                    transformer.get_feature_names_out(cols)
                )
            else:
                feature_names.extend(cols)

        return feature_names


    def gerar_grafico_feature_importance(self, pipeline: Pipeline, X_val: DataFrame, y_val: Series, top_n: int = 20):
        """
        Gera gráfico das top_n features mais importantes do modelo,
        compatível com árvores, linear, SVM e MLP.
        Para SVM e MLP usa Permutation Importance.
        Salva no MLflow como artifact.

        X_val, y_val: dados de validação para calcular permutation importance (para modelos sem coef/feature_importances_)
        """
        model = pipeline.named_steps['regressor']
        preprocessor = pipeline.named_steps['preprocessor']
        feature_names = self.__obter_nomes_features_p(preprocessor)

        importances = None

        # Árvores e ensembles
        if hasattr(model, 'feature_importances_'):
            importances = model.feature_importances_
        # Modelos lineares
        elif hasattr(model, 'coef_'):
            importances = np.abs(model.coef_)
        # SVM e MLP (Permutation Importance)
        else:

            try:
                results = permutation_importance(pipeline, X_val, y_val, n_repeats=10, random_state=42, n_jobs=-1)
                importances = results.importances_mean
            except Exception as e:
                self._logger.warning(f"Não foi possível calcular feature importance: {e}")
                return

        # Ordena as features pelo valor de importância
        sorted_idx = np.argsort(importances)[::-1][:top_n]
        top_features = [feature_names[i] for i in sorted_idx]
        top_importances = importances[sorted_idx]

        # Cria gráfico horizontal
        fig, ax = plt.subplots(figsize=(12, 7))
        if hasattr(model, 'feature_importances_'):
            color = 'skyblue'
        elif hasattr(model, 'coef_'):
            color = 'salmon'
        else:
            color = 'lightgreen'

        ax.barh(range(len(top_features))[::-1], top_importances, color=color)
        ax.set_yticks(range(len(top_features))[::-1])
        ax.set_yticklabels(top_features)
        ax.set_xlabel('Importância da Feature')
        ax.set_title(f'Top {top_n} Features - {model.__class__.__name__}')
        ax.grid(axis='x')
        fig.tight_layout()

        # Salva no buffer e envia para MLflow
        buf = BytesIO()
        fig.savefig(buf, format='png')
        buf.seek(0)
        mlflow.log_image(Image.open(buf), f"plots/feature_importance_{model.__class__.__name__}.png")


