import pandas as pd
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.tree import DecisionTreeClassifier
from sklearn.pipeline import Pipeline
from imblearn.over_sampling import SMOTE
from imblearn.pipeline import Pipeline as IMB_Pipeline
from sklearn.metrics import classification_report


################################################################################################
# Esta função tem como finalidade utilziar o GridSearchCV para encontrar a melhor combinação
# de hiperparâmetros para o modelo de árvore de decisão
# Argumentos de entrada:
#							dados (DataFrame): Dataset completo contendo variáveis preditoras e alvo.
#							alvo: Nome da coluna alvo (y).
#							test_size: Proporção da base usada como teste. Default=0.2
#							random_state: Semente aleatória para reprodutibilidade. Default=42.
#
# Saída:
#							Um objeto do tipo dicionário com os resultados do melhor modelo com 
#							as métricas de desempenho obtidas.
################################################################################################
def treinar_decision_tree_com_gridsearch(dados, alvo, test_size=0.2, random_state=42):
    

    # Treinando o modelo da mesma forma como seria treinado em produção
	#
    X = dados.drop(columns=[alvo])
    y = dados[alvo]
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state, stratify=y
    )
    pipeline = IMB_Pipeline(steps=[
        ('smote', SMOTE(random_state=random_state)),
        ('clf', DecisionTreeClassifier(random_state=random_state))
    ])

    # Selecionando os hiperparâmetros e as faixas que ele deve
	# testar
    param_grid = {
        'clf__criterion': ['gini', 'entropy'],
        'clf__max_depth': [3, 5, 7, 10, 12, 15, 20, 25],
        'clf__min_samples_split': [2, 3, 5, 10, 15, 20],
        'clf__min_samples_leaf': [1, 2, 4, 5, 10, 15, 20]
    }

    # Rodando o GridSearchCV e 
    grid_search = GridSearchCV(
        estimator=pipeline,				# Usando o mesmo recurso de treinar modelo com balanceamento
        param_grid=param_grid,			# Grupo de hiperparâmetros que serão testados
        scoring='f1_macro',   			# Qual métrica será usada para avaliar o desempenho
        cv=5,							# Em quantas partes os dados serão subdivividos para realizar os testes
        n_jobs=-1,						# quantidade de processadores a serem utilizados -1 significa TODOS
        verbose=1						# quantidade de informações que serão exibidas durante a execução 1 mostra o processo de treinamento
    )

    # Treinando
    grid_search.fit(X_train, y_train)

    # Melhor modelo
    best_model = grid_search.best_estimator_

    # Predição no teste
    y_pred = best_model.predict(X_test)

    # Resultados
    resultados = {
        "Melhores Hiperparâmetros": grid_search.best_params_,
        "Melhor Score (CV)": grid_search.best_score_,
        "Relatório de Classificação": classification_report(y_test, y_pred, output_dict=True)
    }

    return resultados

