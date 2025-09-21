import pandas as pd
from sklearn.metrics import confusion_matrix
import numpy as np

##############################################################################################
# Esta função testa o modelo treinado sorteando um número igual de amostras para cada classe 
# e avalia como o modelo acerta.
#
# Argumentos:
#				modelo: O modelo de classificador já treinado (objeto pipeline ou classificador).
#				dataframe (pd.DataFrame): O DataFrame completo contendo features e a variável alvo.
#				variavel_alvo (str): O nome da coluna da variável alvo.
#				n_amostras (int): O número de amostras a serem sorteadas para cada classe.
#
# Saída:		Matriz de confusão do teste
##############################################################################################
def avaliar_modelo_com_amostragem(modelo, dataframe, variavel_alvo, n_amostras):
        
    # Passo 1. Separar features e alvo do dataframe original
    X = dataframe.drop(variavel_alvo, axis=1)
    y = dataframe[variavel_alvo]

    # Passo 2. Identificar as classes únicas
    classes_unicas = y.unique()
    
    # Passo 3. Criar um dataframe para armazenar as amostras sorteadas
    amostra_combinada = pd.DataFrame()

    # Parte 4. Sortear a quantidade de amostras desejada para cada classe
    print(f"Sorteando {n_amostras} amostras para cada classe...")
    for classe in classes_unicas:
        amostras_da_classe = dataframe[dataframe[variavel_alvo] == classe].sample(
            n=n_amostras, random_state=42, replace=False
        )
        amostra_combinada = pd.concat([amostra_combinada, amostras_da_classe], ignore_index=True)
    
    # Parte 5. Separar features e alvo do novo dataframe de amostra
    X_amostra = amostra_combinada.drop(variavel_alvo, axis=1)
    y_amostra = amostra_combinada[variavel_alvo]
    
    # Parte 6. Fazer as predições com o modelo treinado
    y_predito = modelo.predict(X_amostra)
    
    # Parte 7. Criar a matriz de confusão para análise detalhada
    matriz_confusao = confusion_matrix(y_amostra, y_predito)
    
    print("\n--- Resultados da Avaliação por Classe ---")
    
    # Parte 8. Imprimir o número de acertos por classe
    for i, classe in enumerate(classes_unicas):
        # A quantidade de acertos para a classe é o valor na diagonal principal da matriz
        acertos = matriz_confusao[i, i]
        print(f"Classe {classe}: {acertos} acertos de um total de {n_amostras} amostras.")
        
    print("------------------------------------------")
    print("\nMatriz de Confusão:\n", matriz_confusao)
    
    return matriz_confusao