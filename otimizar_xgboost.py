#######################################################################################
# Função para otimização de hiperparâmetros do XGBoost
# Suporta GridSearchCV, RandomizedSearchCV e Optuna
#######################################################################################
import optuna
from sklearn.model_selection import GridSearchCV, RandomizedSearchCV, cross_val_score, train_test_split
from xgboost import XGBClassifier

def otimizar_xgboost(
                        dados,                       # DataFrame completo já tratado
                        variavel_alvo,               # nome da coluna alvo
                        metodo="grid",               # 'grid', 'random', 'optuna'
                        scoring="f1",                # métrica de avaliação
                        cv=3,                        # nº de folds de validação cruzada
                        n_jobs=-1,                   # núcleos paralelos
                        n_iter=20,                   # nº de iterações (random/optuna)
                        param_grid=None,             # dicionário de parâmetros (grid/random)
                        test_size=0.2,               # tamanho do conjunto de teste
                        random_state=42              # semente para reprodutibilidade
                    ):
    """
    Otimiza hiperparâmetros do XGBoost usando GridSearchCV, RandomizedSearchCV ou Optuna.

    Retorna: (modelo_melhor, melhores_parametros, melhor_score)
    """

    # Separar features e alvo
    X = dados.drop(variavel_alvo, axis=1)
    y = dados[variavel_alvo]

    # Divisão treino/teste
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state, stratify=y
    )

    # Modelo base
    xgb = XGBClassifier(
        use_label_encoder=False,
        eval_metric="logloss",
        random_state=random_state
    )

    # Caso 1 - GridSearchCV
    if metodo == "grid":
        if param_grid is None:
            param_grid = {
                'n_estimators': [100, 200],
                'learning_rate': [0.01, 0.1, 0.2],
                'max_depth': [3, 5, 7],
                'subsample': [0.8, 1],
                'colsample_bytree': [0.8, 1]
            }

        grid = GridSearchCV(
            estimator=xgb,
            param_grid=param_grid,
            scoring=scoring,
            cv=cv,
            n_jobs=n_jobs,
            verbose=1
        )
        grid.fit(X_train, y_train)
        return grid.best_estimator_, grid.best_params_, grid.best_score_

    # Caso 2 - RandomizedSearchCV
    elif metodo == "random":
        if param_grid is None:
            param_grid = {
                'n_estimators': [100, 200, 300, 400, 500],
                'learning_rate': [0.01, 0.05, 0.1, 0.2, 0.3],
                'max_depth': [3, 5, 7, 9, 11],
                'min_child_weight': [1, 3, 5, 7],
                'subsample': [0.6, 0.8, 1.0],
                'colsample_bytree': [0.6, 0.8, 1.0],
                'gamma': [0, 0.1, 0.2, 0.3]
            }

        random_search = RandomizedSearchCV(
            estimator=xgb,
            param_distributions=param_grid,
            n_iter=n_iter,
            scoring=scoring,
            cv=cv,
            random_state=random_state,
            n_jobs=n_jobs,
            verbose=1
        )
        random_search.fit(X_train, y_train)
        return random_search.best_estimator_, random_search.best_params_, random_search.best_score_

    # Caso 3 - Optuna
    elif metodo == "optuna":
        def objective(trial):
            params = {
                'n_estimators': trial.suggest_int('n_estimators', 100, 500),
                'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.3, log=True),
                'max_depth': trial.suggest_int('max_depth', 3, 10),
                'min_child_weight': trial.suggest_int('min_child_weight', 1, 10),
                'gamma': trial.suggest_float('gamma', 0, 0.5),
                'subsample': trial.suggest_float('subsample', 0.5, 1.0),
                'colsample_bytree': trial.suggest_float('colsample_bytree', 0.5, 1.0),
                'reg_alpha': trial.suggest_float('reg_alpha', 0, 1.0),
                'reg_lambda': trial.suggest_float('reg_lambda', 0.5, 2.0),
                'n_jobs': n_jobs,
                'random_state': random_state,
                'use_label_encoder': False,
                'eval_metric': 'logloss'
            }

            model = XGBClassifier(**params)
            scores = cross_val_score(model, X_train, y_train, cv=cv, scoring=scoring, n_jobs=n_jobs)
            return scores.mean()

        study = optuna.create_study(direction="maximize")
        study.optimize(objective, n_trials=n_iter)

        best_model = XGBClassifier(**study.best_params)
        best_model.fit(X_train, y_train)

        return best_model, study.best_params, study.best_value

    else:
        raise ValueError("Método inválido. Escolha entre: 'grid', 'random' ou 'optuna'")
