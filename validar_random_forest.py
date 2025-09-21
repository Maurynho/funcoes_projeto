#######################################################################################
# Função para validar um modelo Random Forest já treinado (versão expandida)
#######################################################################################
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    roc_auc_score
)
import numpy as np

def validar_random_forest(modelo, X_test, y_test):
    """
    Valida um modelo RandomForestClassifier já treinado.

    Retorna métricas tradicionais + OOB Score + Estabilidade entre árvores +
    Proximidade entre amostras.
    """

    # Predições do modelo
    y_pred = modelo.predict(X_test)

    resultados = {}
    resultados["Acurácia"] = accuracy_score(y_test, y_pred)
    resultados["Precisão"] = precision_score(y_test, y_pred, average="weighted", zero_division=0)
    resultados["Recall"] = recall_score(y_test, y_pred, average="weighted", zero_division=0)
    resultados["F1-Score"] = f1_score(y_test, y_pred, average="weighted", zero_division=0)
    resultados["Matriz de Confusão"] = confusion_matrix(y_test, y_pred).tolist()

    # AUC (somente para binário)
    if hasattr(modelo, "predict_proba"):
        if len(np.unique(y_test)) == 2:
            y_prob = modelo.predict_proba(X_test)[:, 1]
            resultados["AUC"] = roc_auc_score(y_test, y_prob)

    # OOB Score (se calculado no treino)
    if hasattr(modelo, "oob_score_"):
        resultados["OOB Score"] = modelo.oob_score_

    # Estabilidade entre árvores (variância das previsões entre estimadores)
    if hasattr(modelo, "estimators_"):
        preds_estimadores = np.array([tree.predict(X_test) for tree in modelo.estimators_])
        variancia_por_amostra = np.var(preds_estimadores, axis=0)
        resultados["Estabilidade entre Árvores"] = float(1 - np.mean(variancia_por_amostra))  
        # Quanto mais próximo de 1 → maior estabilidade entre árvores

    # Proximidade entre amostras
    try:
        leaf_indices = [tree.apply(X_test) for tree in modelo.estimators_]
        n_amostras = X_test.shape[0]
        proximidade = np.zeros((n_amostras, n_amostras))

        for leaves in leaf_indices:
            for i in range(n_amostras):
                for j in range(i, n_amostras):
                    if leaves[i] == leaves[j]:
                        proximidade[i, j] += 1
                        if i != j:
                            proximidade[j, i] += 1

        proximidade /= len(modelo.estimators_)
        resultados["Proximidade entre Amostras"] = proximidade.tolist()
    except Exception as e:
        resultados["Proximidade entre Amostras"] = f"Não foi possível calcular: {e}"

    return resultados
