import pandas as pd
from logger import Logger

from static import OUTPUT
from train import train_to_infere
from preprocess import process_dataframes
from train import random_split

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
    x_train, _, y_train, _ = random_split(x_train, y_train, train_size=0.9999)
    model = train_to_infere(x_train, y_train)

    y_hat_test = model.predict(x_test)
    y_test_df: pd.DataFrame = pd.DataFrame(y_hat_test)
    y_test_df.index.name = "Id"
    y_test_df.columns = ["Category"]
    y_test_df.to_csv(f"{OUTPUT}/submission.csv")


############################################################################
# HYPER PARAMETERS EXPLORATION
############################################################################


if __name__ == '__main__':
    main()
