import mlflow

MLFLOW_URI = "http://172.25.0.5:5000"
mlflow.set_tracking_uri(MLFLOW_URI)
mlflow.set_registry_uri(MLFLOW_URI)

# Listar todos os experimentos
experiments = mlflow.search_experiments(view_type=2)  # view_type=2 pega ativos + deletados

for exp in experiments:
    print(f"Deletando permanentemente: ID={exp.experiment_id}, Name={exp.name}")

    # Primeiro, marca como deleted (se ainda não estiver)
    try:
        mlflow.delete_experiment(exp.experiment_id)
    except Exception as e:
        print(f"Erro ao deletar: {e}")

    # Depois, purga permanentemente
    try:
        mlflow.purge_experiment(exp.experiment_id)
    except AttributeError:
        # Em algumas versões do MLflow, purge_experiment não existe
        print("mlflow.purge_experiment() não disponível nesta versão. Verifique a versão do MLflow.")
    except Exception as e:
        print(f"Erro ao purgar: {e}")

print("Todos os experimentos foram removidos permanentemente.")
