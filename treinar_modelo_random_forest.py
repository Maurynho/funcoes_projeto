#######################################################################################
# Função para treinar um modelo utilizando Random Forest
#######################################################################################
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.pipeline import Pipeline

from imblearn.over_sampling import SMOTE
from imblearn.under_sampling import RandomUnderSampler
from imblearn.combine import SMOTEENN
from imblearn.pipeline import Pipeline as IMBPipeline

def treinar_modelo_random_forest(
                                    dados,                              # DataFrame com os dados já tratados
                                    variavel_alvo,                      # variável que se deseja predizer
                                    tecnica_balanceamento="smote",      # técnica de balanceamento ('smote', 'under', 'smoteenn', 'nenhum')
                                    test_size=0.2,                      # tamanho para dados de teste (default 20%)
                                    n_estimators=100,                   # número de árvores na floresta (default 100)
                                    criterio="gini",                    # critério de divisão ('gini', 'entropy', 'log_loss')
                                    max_depth=None,                     # profundidade máxima das árvores
                                    max_features="sqrt",                # nº de features consideradas em cada split ('sqrt' default)
                                    bootstrap=True,                     # usar bootstrap? (default True)
                                    oob_score=False,                    # calcular out-of-bag score? (default False)
                                    class_weight=None,                  # balanceamento de classes ('balanced', dict ou None)
                                    n_jobs=-1,                          # núcleos paralelos (default -1 = todos)
									min_samples_split=2, 
                                    min_samples_leaf=1,
                                    random_state=42                     # semente para reprodutibilidade
                                ):


	#####################################################################################
	# Este algoritmo recebe o dataset completo por isso preciso separar em dados de
	# treino e estes antes de treinar o modelo
	#####################################################################################
    X = dados.drop(variavel_alvo, axis=1)
    y = dados[variavel_alvo]
    X_train, X_test, y_train, y_test = train_test_split(
                                                            X,
                                                            y,
                                                            test_size=test_size,
                                                            random_state=random_state,
                                                            stratify=y
                                                        )
    print("Divisão de dados concluída.")
    print("Distribuição das classes no treino antes do balanceamento:", y_train.value_counts().to_dict())

    
    if tecnica_balanceamento == "smote":
        sampler = SMOTE(random_state=random_state)
    elif tecnica_balanceamento == "under":
        sampler = RandomUnderSampler(random_state=random_state)
    elif tecnica_balanceamento == "smoteenn":
        sampler = SMOTEENN(random_state=random_state)
    elif tecnica_balanceamento == "nenhum":
        sampler = None
    else:
        raise ValueError("Escolha uma técnica de balanceamento válida: 'smote', 'under', 'smoteenn' ou 'nenhum'")

    
    rf_clf = RandomForestClassifier(
                                        n_estimators=n_estimators,
                                        criterion=criterio,
                                        max_depth=max_depth,
                                        max_features=max_features,
                                        bootstrap=bootstrap,
                                        oob_score=oob_score,
                                        class_weight=class_weight,
                                        n_jobs=n_jobs,
										min_samples_split=min_samples_split, 
										min_samples_leaf=min_samples_leaf,
                                        random_state=random_state
                                    )
    
    if sampler:
        modelo_pipeline = IMBPipeline(steps=[
                                                ('sampler', sampler),
                                                ('classifier', rf_clf)
                                            ])
        print(f"Pipeline com {tecnica_balanceamento} e Random Forest criada.")
    else:
        modelo_pipeline = Pipeline(steps=[
            ('classifier', rf_clf)
        ])
        print("Pipeline sem balanceamento e com Random Forest criada.")

    #####################################################################################
	# Treinar e prever 
	#####################################################################################
    modelo_pipeline.fit(X_train, y_train)
    y_pred = modelo_pipeline.predict(X_test)

    return n_s
