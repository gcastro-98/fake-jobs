import pandas as pd
import pickle
from logger import Logger
from static import OUTPUT

logger = Logger()


# MAIN FUNCTIONS


def main() -> None:
    """
    Train the best model, asses the score and return the inferred array
    """
    x: pd.DataFrame = pd.read_csv(f"{OUTPUT}/test.csv", index_col=0)
    with open(f'{OUTPUT}/model.pkl', 'rb') as _model:
        model = pickle.load(_model)
    scaler = pickle.load(open(f'{OUTPUT}/scaler.pkl', 'rb'))
    y_hat_test = model.predict(scaler.transform(x))
    y_test_df: pd.Series = pd.Series(y_hat_test)
    __ammend_dataset(y_test_df)


def __ammend_dataset(dataframe: pd.Series) -> None:
    # dataframe = dataframe.reset_index()
    dataframe.index.name = "Id"
    dataframe.columns = ["Category"]
    dataframe.to_csv("output/submission.csv")


############################################################################
# HYPER PARAMETERS EXPLORATION
############################################################################


if __name__ == '__main__':
    main()
