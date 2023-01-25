from random import randint
import pickle
import numpy as np
from sklearn.metrics import f1_score
from sklearn.model_selection import train_test_split, StratifiedKFold, RandomizedSearchCV
from sklearn.ensemble import RandomForestClassifier
from typing import Tuple, Any
from xgboost import XGBClassifier

from logger import Logger
from static import OUTPUT


logger = Logger()


def train_to_infere(x_train: np.ndarray, y_train: np.ndarray) -> XGBClassifier:
    logger.warning("TRAINING THE MODEL AND SAVING IT FOR INFERENCE")
    # df: pd.DataFrame = pd.read_csv(f"{OUTPUT}/train.csv", index_col=0)
    # x, y = df[[_c for _c in df.columns if _c != LABEL]], df[[LABEL]]

    # scaler = StandardScaler()
    # x_train = scaler.fit_transform(x_train)
    # pickle.dump(scaler, open(f'{OUTPUT}/scaler.pkl', 'wb'))
    model = _train(x_train, y_train)
    pickle.dump(model, open(f'{OUTPUT}/model.pkl', 'wb'))

    return model


def train_to_asses(x: np.ndarray, y: np.ndarray) -> None:
    logger.warning("TRAINING THE MODEL AND ASSESSING THE SKILL")
    # df: pd.DataFrame = pd.read_csv(f"{OUTPUT}/train.csv", index_col=0)
    # x, y = df[[_c for _c in df.columns if _c != LABEL]], df[[LABEL]]
    x_train, x_test, y_train, y_test = random_split(x, y, train_size=0.65)
    model = _train(x_train, y_train)
    metric: float = f1_score(y_test, model.predict(x_test))
    logger.info(f"The performance of the model in the test step is: F1 = {metric}")


def _train(x_train: np.ndarray, y_train: np.ndarray, n: int = 3) -> Any:
    cv_results: dict = {}  # score: model

    logger.info(f"Training in a {n}-StratifiedKFold CV to choose the best one. WAIT")
    skf = StratifiedKFold(n_splits=n, shuffle=True)
    # from sklearn.model_selection import KFold
    # skf = KFold(n_splits=n)
    for _train_ind, _val_ind in skf.split(x_train, y_train):
        model = compile_model(randint(1, 100))
        _x_train, _x_val = x_train[_train_ind], x_train[_val_ind]
        _y_train, _y_val = y_train[_train_ind], y_train[_val_ind]
        try:
            model.fit(_x_train, _y_train.ravel(), eval_set=[(_x_val, _y_val.ravel())], verbose=1)
        except TypeError:  # then it does not contain eval_set as attribute
            model.fit(_x_train, _y_train.ravel())

        # we manually save the model and the score
        _loss: float = f1_score(_y_val.ravel(), model.predict(_x_val))
        cv_results[_loss] = model
        logger.debug(f"\tModel trained with validation F1 = {_loss}")
    logger.debug(f"\tTraining finished with validations F1: "
                 f"{', '.join([str(_s) for _s in cv_results.keys()])}")

    _sorted_cv_dict: dict = dict(
        sorted(cv_results.items(), key=lambda x: x[0], reverse=True))

    logger.info(f"Selecting the model with F1 = {list(_sorted_cv_dict.keys())[0]}")
    return list(_sorted_cv_dict.values())[0]


def compile_model(random_state: int) -> Any:
    # if _rf:
    #    return RandomForestClassifier(max_depth=5, random_state=random_state, n_estimators=1500)

    best_params = {
        'subsample': 0.8, 'n_estimators': 1500, 'min_child_weight': 12,
        'max_depth': 6, 'learning_rate': 0.05, 'gamma': 1, 'colsample_bytree': 0.8}
    best_params.update(
        {'objective': 'binary:logistic', 'nthread': -1, 'seed': random_state,
         'eval_metric': 'auc', 'use_label_encoder': False, 'early_stopping_rounds': 50})
    model = XGBClassifier(**best_params)
    # logger.debug("\tCompiling an XGB regressor with the following hyperparameters: {best_params.__str__()}")

    # model = RandomForestClassifier(max_depth=30, n_estimators=2000, random_state=random_state)
    return model


############################################################################################
# AUXILIARY
############################################################################################

def random_split(features: np.ndarray, labels: np.ndarray,
                 train_size: float = 0.75) \
        -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    logger.debug(f"\t Splitting (using {np.round(train_size, 2)}, "
                 f"{np.round(1-train_size, 2)} split) and scaling the data.")
    x_train, x_test, y_train, y_test = train_test_split(
        features, labels, train_size=train_size, shuffle=True)
    # scaler: StandardScaler = StandardScaler()
    # x_train = scaler.fit_transform(x_train)
    # x_test = scaler.fit_transform(x_test)
    # pickle.dump(scaler, open(f'{OUTPUT}/scaler.pkl', 'wb'))
    return x_train, x_test, y_train, y_test


############################################################################################
# MODEL FINE-TUNING
############################################################################################

def __random_search(x_train: np.ndarray, y_train: np.ndarray) -> None:
    """
    Parameter grid search for XGBoost
    """
    params, model = __define_params_and_model('xgb')

    skf = StratifiedKFold(n_splits=2, shuffle=False)
    random_search = RandomizedSearchCV(
        model, param_distributions=params, n_iter=10, scoring='f1',
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


def __define_params_and_model(arquitecture: str):
    if arquitecture == 'xgb':
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
        model = XGBClassifier(**{'objective': 'binary:logistic', 'eval_metric': 'auc', 'use_label_encoder': False})

    elif arquitecture == 'rf':
        # We define some parameters to explore
        params = {
            'n_estimators': [200, 500, 800, 1200, 1600, 1800, 2500, 5000],
            'max_depth': [2, 3, 5, 8]
        }
        model = RandomForestClassifier()

    else:
        raise KeyError(f"Unrecognized arquitecture: {arquitecture}")
    return params, model


if __name__ == '__main__':
    from preprocess import process_dataframes
    __x_train, __y_train, _ = process_dataframes()
    # __random_search(__x_train, __y_train)
    # train_to_infere(__x_train, __y_train)
    train_to_asses(__x_train, __y_train)
