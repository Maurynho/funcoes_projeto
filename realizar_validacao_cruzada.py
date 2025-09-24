import numpy as np
from sklearn.model_selection import cross_validate
from sklearn.metrics import make_scorer, accuracy_score, precision_score, recall_score, f1_score, roc_auc_score

##############################################################################################
#    Realiza a validação cruzada em um modelo de classificação e imprime as métricas.
#
#    Argumentos:
#        modelo: Um classificador compatível com scikit-learn (ex: RandomForestClassifier, XGBClassifier).
#        X (DataFrame): Variáveis preditoras.
#        y (Series): Variável alvo.
#        cv (int): Número de folds da validação cruzada. Default é 5.
#
#    Retorna:
#        Um dicionário com as métricas de desempenho detalhadas.
##############################################################################################
def realizar_validacao_cruzada(modelo, X, y, cv=5):
    
    print("Iniciando a validação cruzada...")
    print(f"Modelo: {modelo.__class__.__name__}")
    print(f"Número de folds (cv): {cv}\n")
    
    # Detecta se é problema binário ou multiclasses
    n_classes = len(np.unique(y))
    needs_proba = True  # Para ROC AUC precisamos de probabilidades
    roc_auc_kwargs = {}
    if n_classes > 2:
        roc_auc_kwargs['multi_class'] = 'ovr'
    
    # Definindo as métricas
    scorers = {
        'acuracia': make_scorer(accuracy_score),
        'precisao': make_scorer(precision_score, average='macro', zero_division=0),
        'recall': make_scorer(recall_score, average='macro', zero_division=0),
        'f1_score': make_scorer(f1_score, average='macro', zero_division=0),
        'auc_roc': make_scorer(roc_auc_score, needs_proba=needs_proba, **roc_auc_kwargs)
    }

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
