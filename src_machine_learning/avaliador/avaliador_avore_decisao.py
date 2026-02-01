import logging
from datetime import datetime
from typing import Dict, Any, List, Optional

import numpy as np
from matplotlib import pyplot as plt
from pandas import Series, DataFrame
from sklearn.metrics import mean_absolute_error, mean_squared_error, median_absolute_error, r2_score
from sklearn.model_selection import GridSearchCV, validation_curve
from sklearn.pipeline import Pipeline
from sklearn.tree import DecisionTreeRegressor, export_text

from src_machine_learning.avaliador.avaliador import Avaliador


class AvaliadorArvoreDecisao(Avaliador):
    def __init__(self):
        self.__param_range = np.arange(2, 50, 2)
        self._logger = logging.getLogger(self.__class__.__name__)

    def obter_dados_curva_validacao(self, pipeline: Pipeline, X_train: DataFrame, y_train: Series) -> Dict[str, Any]:
        train_scores, val_scores = validation_curve(
            estimator=pipeline,
            X=X_train,
            y=y_train,
            param_name='regressor__max_depth',
            param_range=self.__param_range,
            cv=5,
            scoring='neg_root_mean_squared_error',
            n_jobs=-1
        )

        train_rmse = -train_scores.mean(axis=1)
        val_rmse = -val_scores.mean(axis=1)

        train_std = train_scores.std(axis=1)
        val_std = val_scores.std(axis=1)

        best_idx = np.argmin(val_rmse)
        best_depth = float(self.__param_range[best_idx])
        best_rmse = float(val_rmse[best_idx])

        resultados = {
            'train_rmse': train_rmse.tolist(),
            'val_rmse': val_rmse.tolist(),
            'train_std': train_std.tolist(),
            'val_std': val_std.tolist(),
            'best_depth': best_depth,
            'best_rmse': best_rmse

        }
        return resultados

    @staticmethod
    def __obter_nomes_features(preprocessor) -> List[str]:
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

    def __gerar_regras_arvore(
            self,
            tree: DecisionTreeRegressor,
            pipeline: Pipeline,
            feature_names: List[str],
            max_depth: Optional[int] = None,

    ) -> str:
        assert pipeline is not None, "Pipeline não foi treinado"

        regras = export_text(
            decision_tree=tree,
            feature_names=feature_names,
            max_depth=max_depth,
            decimals=3
        )

        self._logger.info("=" * 80)
        self._logger.info("REGRAS DA ÁRVORE DE DECISÃO")
        self._logger.info("Profundidade máxima exibida: %s", max_depth)
        self._logger.info("=" * 80)

        for linha in regras.split("\n"):
            self._logger.info(linha)

        self._logger.info("=" * 80)
        return regras

    def obter_resultados_modelo(self, pipeline: Pipeline, y_test: Series, y_pred:  np.ndarray[Any, np.dtype[Any]]) -> Dict[str, Any]:
        assert pipeline is not None, "Pipeline não foi treinado"

        tree: DecisionTreeRegressor = pipeline.named_steps['regressor']
        preprocessor = pipeline.named_steps['preprocessor']

        y_test_arr = np.asarray(y_test)
        y_pred_arr = np.asarray(y_pred)

        mae = mean_absolute_error(y_test_arr, y_pred_arr)
        rmse = np.sqrt(mean_squared_error(y_test_arr, y_pred_arr))
        medae = median_absolute_error(y_test_arr, y_pred_arr)
        r2 = r2_score(y_test_arr, y_pred_arr)

        # SMAPE (robusto para imóveis)
        smape = np.mean(
            2.0 * np.abs(y_pred_arr - y_test_arr)
            / (np.abs(y_test_arr) + np.abs(y_pred_arr))
        ) * 100

        # Bias (viés do modelo)
        bias = float(np.mean(y_pred_arr - y_test_arr))

        # % de previsões com erro ≤ 10%
        erro_percentual = np.abs(y_pred_arr - y_test_arr) / y_test_arr
        acc_10 = float(np.mean(erro_percentual <= 0.10))

        feature_names = self.__obter_nomes_features(preprocessor)
        regras_arvore = self.__gerar_regras_arvore(
            tree=tree,
            feature_names=feature_names,
            max_depth=tree.get_depth(),
            pipeline=pipeline,
        )

        feature_importances = dict(
            sorted(
                zip(
                    feature_names,
                    map(float, tree.feature_importances_)
                ),
                key=lambda x: x[1],
                reverse=True
            )
        )

        resultados: Dict[str, Any] = {
            # Métricas principais
            "mae": mae,
            "rmse": float(rmse),
            "medae": medae,
            "smape": float(smape),
            "r2": r2,
            "bias": bias,
            "accuracy_erro_10_pct": acc_10,

            # Estatísticas do target
            "preco_medio_real": float(np.mean(y_test_arr)),
            "preco_medio_previsto": float(np.mean(y_pred_arr)),
            "n_amostras": len(y_test_arr),

            # Complexidade do modelo
            "profundidade_arvore": tree.get_depth(),
            "numero_folhas": int(tree.get_n_leaves()),
            "numero_nos": tree.tree_.node_count,

            # Interpretabilidade
            "feature_importances": feature_importances,
            # "regras_arvore": regras_arvore
        }

        return resultados

    def gerar_grafico_underfit_overfit(self, dados: Dict[str, Any]):
        plt.figure(figsize=(12, 7))
        param_range = self.__param_range
        train_rmse = dados['train_rmse']
        val_rmse = dados['val_rmse']
        best_depth = dados['best_depth']
        best_rmse = dados['best_rmse']

        # Curvas
        plt.plot(
            param_range,
            train_rmse,
            marker='o',
            linewidth=2,
            label='RMSE Treino (Viés)'
        )

        plt.plot(
            param_range,
            val_rmse,
            marker='o',
            linewidth=2,
            label='RMSE Validação (Generalização)'
        )

        # Gap (variância)
        plt.fill_between(
            param_range,
            train_rmse,
            val_rmse,
            alpha=0.2,
            label='Gap Treino × Validação (Variância)'
        )

        # Linha vertical no melhor depth
        plt.axvline(
            x=best_depth,
            linestyle='--',
            linewidth=2,
            label=f'Melhor max_depth = {best_depth}'
        )

        # Marcar ponto ótimo
        plt.scatter(
            best_depth,
            best_rmse,
            s=120,
            zorder=5
        )

        # Anotações
        plt.text(
            best_depth + 0.5,
            best_rmse * 1.03,
            f'Mínimo RMSE (CV)\nRMSE ≈ {best_rmse:,.0f}',
            fontsize=10
        )

        plt.text(
            param_range[0],
            max(val_rmse) * 0.95,
            'UNDERFITTING\n(alto viés)',
            fontsize=11,
            ha='left'
        )

        plt.text(
            param_range[-1],
            min(train_rmse) * 1.05,
            'OVERFITTING\n(alta variância)',
            fontsize=11,
            ha='right'
        )

        # Labels finais
        plt.xlabel('max_depth (Complexidade do Modelo)')
        plt.ylabel('RMSE')
        plt.title('Decision Tree — Diagnóstico de Overfitting vs Underfitting')
        plt.legend()
        plt.grid(True)
        plt.tight_layout()
        plt.savefig(
            f'fig/gerar_grafico_over_under/arvore_decisao/gerar_grafico_underfit_overfit_over_under_av_{datetime.now().strftime("%Y_%m_%d__%H_%M_%S")}.png')
        plt.close()

    def obter_resultado_grid_search(self, grid_search: GridSearchCV) ->  Dict[str, Any]:
        best_params = grid_search.best_params_
        best_estimator = grid_search.best_estimator_
        best_score = grid_search.best_score_
        rmse = abs(grid_search.best_score_)

        # Função para converter tipos do NumPy para nativos

        def to_native(val):
            if isinstance(val, (np.integer, np.int64)):
                return int(val)
            if isinstance(val, (np.floating, np.float64)):
                return float(val)
            if isinstance(val, np.ndarray):
                return val.tolist()
            return val

            # Converte todos os valores de best_params

        best_params_native = {k: to_native(v) for k, v in best_params.items()}
        best_score_native = float(best_score)  # Garantir tipo nativo float

        return {
            'best_params': best_params_native,
            'best_score': best_score_native,
            'rmse': float(rmse)
        }
