#######################################################################################
# Função para validar um modelo XGBoost já treinado
#######################################################################################
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    roc_auc_score,
    log_loss
)
import numpy as np

def validar_modelo_xgboost(modelo, X_test, y_test):
    """
    Valida um modelo XGBoost já treinado.

    Retorna métricas tradicionais + LogLoss + Importância das variáveis.
    """

    # Predições
    y_pred = modelo.predict(X_test)

    resultados = {}
    resultados["Acurácia"] = accuracy_score(y_test, y_pred)
    resultados["Precisão"] = precision_score(y_test, y_pred, average="weighted", zero_division=0)
    resultados["Recall"] = recall_score(y_test, y_pred, average="weighted", zero_division=0)
    resultados["F1-Score"] = f1_score(y_test, y_pred, average="weighted", zero_division=0)
    resultados["Matriz de Confusão"] = confusion_matrix(y_test, y_pred).tolist()

    # Probabilidades (se disponíveis)
    if hasattr(modelo, "predict_proba"):
        y_prob = modelo.predict_proba(X_test)
        if len(np.unique(y_test)) == 2:
            resultados["AUC"] = roc_auc_score(y_test, y_prob[:, 1])
        resultados["LogLoss"] = log_loss(y_test, y_prob)

    # Importância das variáveis
    try:
        importancia = modelo.get_booster().get_score(importance_type="gain")
        resultados["Importância das Variáveis (gain)"] = importancia
    except Exception as e:
        resultados["Importância das Variáveis"] = f"Não foi possível calcular: {e}"

    # Número de árvores do ensemble
    try:
        resultados["Número de Árvores"] = modelo.get_booster().best_iteration + 1
    except:
        try:
            resultados["Número de Árvores"] = len(modelo.get_booster().get_dump())
        except:
            resultados["Número de Árvores"] = "Não disponível"

    return resultados
