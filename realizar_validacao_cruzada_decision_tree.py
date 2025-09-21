import numpy as np
from sklearn.model_selection import cross_validate
from sklearn.metrics import make_scorer, accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
from sklearn.tree import DecisionTreeClassifier
from sklearn.datasets import make_classification


def realizar_validacao_cruzada_decision_tree(dados, alvo, modelo, cv=5):
    """
    Realiza a validação cruzada em um modelo e imprime as métricas de desempenho.

    Argumentos:
        modelo: Um modelo de classificador do scikit-learn (ex: DecisionTreeClassifier).
        X (DataFrame ou array): Variáveis preditoras.
        y (Series ou array): Variável alvo.
        cv (int): O número de 'folds' para a validação cruzada. Default é 5.

    Retorna:
        Um dicionário com as métricas de desempenho detalhadas.
    """
    
    print("Iniciando a validação cruzada...")
    print(f"Modelo: {modelo.__class__.__name__}")
    print(f"Número de folds (cv): {cv}\n")

    # Definindo as métricas para avaliação
    scorers = {
                  'acuracia': make_scorer(accuracy_score),
                  'precisao': make_scorer(precision_score, average='macro', zero_division=0),
                  'recall': make_scorer(recall_score, average='macro', zero_division=0),
                  'f1_score': make_scorer(f1_score, average='macro', zero_division=0),
                  'auc_roc': make_scorer(roc_auc_score)
              }
	
    X = dados.drop(alvo, axis=1)
    y = dados[alvo]
    print("Variáveis X e y obtidas do DataFrame original.")

    # Realizando a validação cruzada
    resultados = cross_validate(
        estimator=modelo,
        X=X,
        y=y,
        cv=cv,
        scoring=scorers,
        return_train_score=False,
        n_jobs=-1
    )

    # Imprimindo os resultados
    print("--- Resultados da Validação Cruzada ---")
    
    for metrica in scorers.keys():
        scores = resultados[f'test_{metrica}']
        print(f"{metrica.upper():<12} | Média: {np.mean(scores):.4f} | Desvio Padrão: {np.std(scores):.4f}")
        print(f"{'':<12} | Scores por fold: {[f'{s:.4f}' for s in scores]}\n")

    print("---------------------------------------")

    return resultados
