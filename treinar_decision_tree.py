#######################################################################################
# Importações necessárias
#######################################################################################
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.pipeline import Pipeline
import matplotlib.pyplot as plt

from imblearn.over_sampling import SMOTE
from imblearn.under_sampling import RandomUnderSampler
from imblearn.combine import SMOTEENN
from imblearn.pipeline import Pipeline as IMBPipeline # Importa a pipeline do imblearn

#######################################################################################
# A função treinar_decision_tree será tem como finalidade estabelecer um pipeline
# balanceamento -> treino -> avaliação de Decision Tree
# Parâmetros:
#               - dados(DataFrame): Dataset completo com features e alvo
#               - variavel_alvo (str): Nome da coluna alvo
#               - tecnica_balanceamento (str): 'smote', 'under', 'smoteenn' ou 'nenhum'.
#					- Se optar por nenhum será aplicado op parâmetro class_weight 
#               - test_size (float): Proporção do tamanho dos dados de teste (default 0.2 = 20%)
#               - profundidade (int): Profundidade máxima da árvore (default 5)
#               - criterio (str): define a métrica de critério de treino. Valores "gini"(default) ou "entropy"
#
# Retorno: a saída desta função são as variáveis resultantes do treinamento do modelo
#               - pipeline treinado
#               - X_test
#               - y_test
#               - y_pred
#######################################################################################
def treinar_modelo_arvore(
										dados,								# DataFrame com os dados já tratados
										variavel_alvo,						# variável que se deseja predizer
										tecnica_balanceamento="smote",		# seleção da técnica de balanceamento com default em SMOTE
										test_size=0.2,						# tamanho para dados de teste - default 20%
										profundidade=5,						# produndidade da árvore - default 5
										criterio="gini",					# critério de de treino - default índice de gini
										random_state=42,					# semente do gerador de números aleatórios - default 42
										min_samples_split=2,				# mínimo de amostras necessário para dividir um nó - default 2
										min_samples_leaf=1					# mínimo de amostras que um nó folha deve conter - default 1
									):

    # Fase 1 - Separar features e alvo
    X = dados.drop(variavel_alvo, axis=1)
    y = dados[variavel_alvo]

    # Fase 2 - Separar dados de treino e de teste (dados brutos, antes do balanceamento)
    X_train, X_test, y_train, y_test = train_test_split(
															X,
															y,
															test_size=test_size,
															random_state=random_state,
															stratify=y
														)
    print("Divisão de dados concluída.")
    print("Distribuição das classes no treino antes do balanceamento:", y_train.value_counts().to_dict())

    # Fase 3 - Definir a técnica de balanceamento (sampler)
    # A pipeline do imblearn permite que a etapa de balanceamento seja tratada
    # como parte do treinamento, evitando vazamento de dados (data leakage).
    if tecnica_balanceamento == "smote":
        sampler = SMOTE(random_state=random_state)
    elif tecnica_balanceamento == "under":
        sampler = RandomUnderSampler(random_state=random_state)
    elif tecnica_balanceamento == "smoteenn":
        sampler = SMOTEENN(random_state=random_state)
    elif tecnica_balanceamento == "nenhum":
        sampler = None  # Não aplica balanceamento
    else:
        raise ValueError("Escolha uma técnica de balanceamento válida: 'smote', 'under', 'smoteenn' ou 'nenhum'")


    # Fase 4 - Construir o Pipeline usando IMBPipeline
    # A pipeline garante que o balanceamento será aplicado APENAS nos dados de treino.
    if sampler:
        # Usa a pipeline do imblearn que é compatível com os samplers
        modelo_pipeline = IMBPipeline(steps=[
												('sampler', sampler),
												('classifier', DecisionTreeClassifier(
													criterion=criterio,
													max_depth=profundidade,
													random_state=random_state,
													min_samples_split=min_samples_split,
													min_samples_leaf=min_samples_leaf
												))
											])
        print(f"Pipeline com {tecnica_balanceamento} e Decision Tree criada.")
    else:
        # Se não houver sampler, a pipeline padrão do sklearn é suficiente
        modelo_pipeline = Pipeline(steps=[
            ('classifier', DecisionTreeClassifier(
													criterion=criterio,
													max_depth=profundidade,
													random_state=random_state,
													min_samples_split=min_samples_split,
													min_samples_leaf=min_samples_leaf,
													class_weight='balanced'				#Forçar o balanceamento durante o treinamento
												)
			)
												
        ])
        print("Pipeline sem balanceamento e com Decision Tree criada.")

    # Fase 5 - Treinar o modelo e fazer as predições
    # O pipeline cuidará de aplicar o 'sampler' apenas no X_train/y_train
    modelo_pipeline.fit(X_train, y_train)
    y_pred = modelo_pipeline.predict(X_test)

    return modelo_pipeline, X_test, y_test, y_pred