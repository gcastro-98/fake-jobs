import pandas as pd
from logger import Logger

from static import OUTPUT
from train import train_to_infere
from preprocess import process_dataframes
from train import random_split
from sklearn.metrics import f1_score

logger = Logger()


# MAIN FUNCTIONS


def main() -> None:
    """
    Train the best model, asses the score and return the inferred array
    """
    # with open(f'{OUTPUT}/model.pkl', 'rb') as _model:
    #     model = pickle.load(_model)
    x_train, y_train, x_test = process_dataframes()
    # we perform the split so that the sparse matrix are amended to arrays
    x_train, _x_inner_test, y_train, _y_inner_test = random_split(x_train, y_train, train_size=0.8)
    model = train_to_infere(x_train, y_train)
    print("Inner test score evaluation: ", f1_score(_y_inner_test, model.predict(_x_inner_test)))

    y_hat_test = model.predict(x_test)
    y_test_df: pd.DataFrame = pd.DataFrame(y_hat_test)
    y_test_df.to_csv(f"{OUTPUT}/raw_submission.csv")
    __ammend_dataset(y_test_df)


def __ammend_dataset(dataframe: pd.DataFrame) -> None:
    # dataframe = dataframe.reset_index()
    dataframe.index.name = "Id"
    dataframe.columns = ["Category"]
    dataframe.to_csv(f"{OUTPUT}/submission.csv")


############################################################################
# HYPER PARAMETERS EXPLORATION
############################################################################


if __name__ == '__main__':
    main()
