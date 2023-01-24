from random import randint
import pickle
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import f1_score
from sklearn.model_selection import train_test_split, StratifiedKFold, RandomizedSearchCV
from sklearn.preprocessing import StandardScaler
from typing import Tuple, Any
from xgboost import XGBClassifier

from logger import Logger
from static import OUTPUT, LABEL


logger = Logger()


def main_for_prediction() -> None:
    df: pd.DataFrame = pd.read_csv(f"{OUTPUT}/train.csv", index_col=0)
    x, y = df[[_c for _c in df.columns if _c != LABEL]], df[[LABEL]]
    scaler = StandardScaler()
    x_train, y_train = scaler.fit_transform(x.values), y.values
    pickle.dump(scaler, open(f'{OUTPUT}/scaler.pkl', 'wb'))
    model = _train(x_train, y_train)
    pickle.dump(model, open(f'{OUTPUT}/model.pkl', 'wb'))


def main_for_assesment() -> None:
    df: pd.DataFrame = pd.read_csv(f"{OUTPUT}/train.csv", index_col=0)
    x, y = df[[_c for _c in df.columns if _c != LABEL]], df[[LABEL]]
    x_train, x_test, y_train, y_test = _split_and_scale(x, y)
    model = _train(x_train, y_train)

    y_hat_test = model.predict(x_test)
    metric: float = f1_score(y_test, y_hat_test)
    logger.info(f"The performance of the model in the test step is: F1 = {metric}")


def _train(x_train: np.ndarray, y_train: np.ndarray, n: int = 5) -> Any:
    cv_results: dict = {}  # score: model

    logger.info(f"Training in a {n}-StratifiedKFold CV to choose the best one. WAIT")
    skf = StratifiedKFold(n_splits=n, shuffle=True)
    for _train_ind, _val_ind in skf.split(x_train, y_train):
        model = compile_model(randint(1, 100))
        _x_train, _x_val = x_train[_train_ind], x_train[_val_ind]
        _y_train, _y_val = y_train[_train_ind], y_train[_val_ind]
        model.fit(_x_train, _y_train.ravel(), eval_set=[(_x_val, _y_val.ravel())], verbose=0)

        # we manually save the model and the score
        _loss: float = model.score(_x_val, _y_val.ravel())
        cv_results[_loss] = model
    logger.debug(f"\tTraining finished with validation losses: "
                 f"{', '.join([str(_s) for _s in cv_results.keys()])}")

    _sorted_cv_dict: dict = dict(
        sorted(cv_results.items(), key=lambda x: x[0], reverse=True))
    return list(_sorted_cv_dict.values())[0]


def compile_model(random_state: int, _rf: bool = False) -> Any:
    if _rf:
        return RandomForestClassifier(max_depth=8, random_state=random_state, n_estimators=1500)

    best_params = {
        'subsample': 1.0, 'n_estimators': 1500, 'min_child_weight': 12,
        'max_depth': 5, 'learning_rate': 0.02, 'gamma': 2, 'colsample_bytree': 0.9}

    best_params.update(
        {'objective': 'binary:logistic', 'nthread': -1, 'seed': random_state,
         'eval_metric': 'auc', 'use_label_encoder': False})
    model = XGBClassifier(**best_params)
    logger.debug("\tCompiling an XGB regressor with the following"
                 f" hyper-parameters: {best_params.__str__()}")
    return model


############################################################################################
# AUXILIARY
############################################################################################

def _split_and_scale(features: pd.DataFrame, labels: pd.DataFrame,
                     train_size: float = 0.75) \
        -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    logger.debug(f"\t Splitting (using {np.round(train_size, 2)}, "
                 f"{np.round(1-train_size, 2)} split) and scaling the data.")
    x_train, x_test, y_train, y_test = train_test_split(
        features.values, labels.values, train_size=train_size, shuffle=True)
    scaler: StandardScaler = StandardScaler()
    x_train = scaler.fit_transform(x_train)
    x_test = scaler.fit_transform(x_test)

    pickle.dump(scaler, open(f'{OUTPUT}/scaler.pkl', 'wb'))
    return x_train, x_test, y_train, y_test


############################################################################################
# MODEL FINE-TUNING
############################################################################################

def __random_search() -> None:
    """
    Parameter grid search for XGBoost
    """
    df: pd.DataFrame = pd.read_csv(f"{OUTPUT}/train.csv", index_col=0)
    x, y = df[[_c for _c in df.columns if str(_c) != LABEL]], df[[LABEL]]
    x_train, y_train = StandardScaler().fit_transform(x.values), y.values

    # We define some parameters to explore
    params = {
        'learning_rate': [0.02, 0.01, 0.005],
        'n_estimators': [1200, 1500, 1800, 2100, 2400],
        'min_child_weight': [10, 12, 15, 18, 21],
        'gamma': [1, 2, 5],
        'subsample': [0.8, 0.9, 1.],
        'colsample_bytree': [0.8, 0.9, 1.],
        'max_depth': [2, 3, 5]
    }

    xgb = XGBClassifier(**{'objective': 'binary:logistic', 'eval_metric': 'auc', 'use_label_encoder': False})
    skf = StratifiedKFold(n_splits=2, shuffle=False)
    random_search = RandomizedSearchCV(
        xgb, param_distributions=params, n_iter=10, scoring='f1',
        verbose=2, cv=skf.split(x_train, y_train))
    # scoring metrics: https://scikit-learn.org/stable/modules/model_evaluation.html#scoring-parameter
    logger.info("WAIT: this may take a while")
    random_search.fit(x_train, y_train.ravel())
    # summarize performance
    logger.info('Best estimator:')
    logger.info(random_search.best_estimator_.__str__())
    logger.info('Best hyper-parameters:')
    logger.info(random_search.best_params_.__str__())
    logger.debug("Saving all results")
    # results = pd.DataFrame(random_search.cv_results_)
    # results.to_csv(f'{OUTPUT}/search_results.csv', index=False)


if __name__ == '__main__':
    main_for_prediction()
    # main_for_assesment()
    # __random_search()
