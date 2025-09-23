#######################################################################################
# Função para treinar um modelo utilizando XGBoost
#######################################################################################
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.pipeline import Pipeline

from imblearn.over_sampling import SMOTE
from imblearn.under_sampling import RandomUnderSampler
from imblearn.combine import SMOTEENN
from imblearn.pipeline import Pipeline as IMBPipeline

from xgboost import XGBClassifier

def treinar_modelo_xgboost(
                            dados,                              # DataFrame com os dados já tratados
                            variavel_alvo,                      # variável que se deseja predizer
                            tecnica_balanceamento="smote",      # técnica de balanceamento ('smote', 'under', 'smoteenn', 'nenhum')
                            test_size=0.2,                      # tamanho para dados de teste (default 20%)
                            n_estimators=100,                   # número de árvores (default 100)
                            learning_rate=0.1,                  # taxa de aprendizado (default 0.1)
                            max_depth=6,                        # profundidade máxima das árvores (default 6)
                            min_child_weight=1,                 # peso mínimo em um nó folha
                            gamma=0,                            # complexidade mínima para split adicional
                            subsample=1,                        # fração de amostras usadas em cada árvore
                            colsample_bytree=1,                 # fração de features usadas em cada árvore
                            reg_alpha=0,                        # regularização L1
                            reg_lambda=1,                       # regularização L2
                            scale_pos_weight=1,                 # peso para classes desbalanceadas
                            n_jobs=-1,                          # núcleos paralelos (default -1 = todos)
                            random_state=42                     # semente para reprodutibilidade
                        ):
    #####################################################################################
    # Separar dados em treino e teste
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

    #####################################################################################
    # Definir técnica de balanceamento
    #####################################################################################
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

    #####################################################################################
    # Instanciar classificador XGBoost
    #####################################################################################
    xgb_clf = XGBClassifier(
                                n_estimators=n_estimators,
                                learning_rate=learning_rate,
                                max_depth=max_depth,
                                min_child_weight=min_child_weight,
                                gamma=gamma,
                                subsample=subsample,
                                colsample_bytree=colsample_bytree,
                                reg_alpha=reg_alpha,
                                reg_lambda=reg_lambda,
                                scale_pos_weight=scale_pos_weight,
                                n_jobs=n_jobs,
                                random_state=random_state,
                                use_label_encoder=False,
                                eval_metric="logloss"
                            )

    if sampler:
        modelo_pipeline = IMBPipeline(steps=[
                                                ('sampler', sampler),
                                                ('classifier', xgb_clf)
                                            ])
        print(f"Pipeline com {tecnica_balanceamento} e XGBoost criada.")
    else:
        modelo_pipeline = Pipeline(steps=[
            ('classifier', xgb_clf)
        ])
        print("Pipeline sem balanceamento e com XGBoost criada.")

    #####################################################################################
    # Treinar e prever 
    #####################################################################################
    modelo_pipeline.fit(X_train, y_train)
    y_pred = modelo_pipeline.predict(X_test)

    return modelo_pipeline, X_test, y_test, y_pred
