#######################################################################################
# Importar todas as bibliotecas usadas pela função
#######################################################################################
from sklearn.metrics import (       ## Recursos para avaliar as métricas
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    roc_auc_score,
    mean_squared_error,
    mean_absolute_error,
    r2_score
)
import numpy as np

#######################################################################################
#   Esta função foi implementada para validar apenas modelo de classificação
#
#    Parâmetros:
#                   - modelo : modelo treinado (scikit-learn ou similar)
#                   - X_test : features do conjunto de teste
#                   - y_test : valores reais do teste
#                   - y_pred : previsões do modelo
#
#    Retorno:
#                   - dicionário com métricas calculadas
#######################################################################################
def validar_modelo(modelo, X_test, y_test, y_pred):

    resultados = {} # Dicionário com dados de validação

    # Métricas para classificação
    resultados["Acurácia"] = accuracy_score(y_test, y_pred)
    resultados["Precisão"] = precision_score(y_test, y_pred, average="weighted", zero_division=0)
    resultados["Recall"] = recall_score(y_test, y_pred, average="weighted", zero_division=0)
    resultados["F1-Score"] = f1_score(y_test, y_pred, average="weighted", zero_division=0)
    resultados["Matriz de Confusão"] = confusion_matrix(y_test, y_pred).tolist()

    # Se o modelo suportar predict_proba, calcula AUC
    if hasattr(modelo, "predict_proba"):
        y_prob = modelo.predict_proba(X_test)[:, 1] if len(np.unique(y_test)) == 2 else None
        if y_prob is not None:
            resultados["AUC"] = roc_auc_score(y_test, y_prob)

    return resultados
