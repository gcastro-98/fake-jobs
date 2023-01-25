import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from logger import Logger
# from sklearn.decomposition import PCA
from typing import List, Dict, Tuple
from warnings import catch_warnings, simplefilter
from static import INPUT, EMBEDDED_FEATURES, BINARY_NAN_FEATURES, ONEHOT_FEATURES, \
    NUMERICAL_FEATURE, SYNTHETIC_FEATURE, LABEL, OUTPUT
from scipy.sparse import hstack
import nltk

nltk.download('stopwords')
stop = nltk.corpus.stopwords.words('english')
vectorizers: Dict[str, TfidfVectorizer] = {}
logger = Logger()


#######################################################################
# MAIN
#######################################################################

def process_dataframes(_save_data: bool = False) -> Tuple[np.ndarray, ...]:
    ################################################################
    # TRAIN DATAFRAME
    ################################################################

    logger.info(f"Loading the input train dataset")
    train: pd.DataFrame = __reduce_mem_usage(pd.read_csv(
        f'{INPUT}/train.csv', sep=',', header=0, index_col=0))

    # NECESSARY PREPROCESSING (nan removal & encoding into strings)
    train = basic_preprocessing(train)
    train.drop(labels=BINARY_NAN_FEATURES, axis=1, inplace=True)

    # TEXT EMBEDDINGS
    x_train_df = train.drop(labels=LABEL, axis=1)
    x_train = embed_text_features_separately(x_train_df, columns=EMBEDDED_FEATURES + ONEHOT_FEATURES, train=True)
    y_train = train[LABEL].values

    # pca = PCA(n_components=x_train.shape[0] // 2)
    # x_train = pca.fit_transform(x_train.toarray())

    # TRAIN DATA SERIALIZATION
    if _save_data:
        logger.debug("\tSerializing the train data")
        with open(f'{OUTPUT}/train.npz', 'wb') as train_outfile:
            np.savez(train_outfile,
                     x=x_train, y=y_train)

    ################################################################
    # TEST DATAFRAME
    ################################################################

    logger.info(f"Loading the input test dataset")
    test: pd.DataFrame = __reduce_mem_usage(pd.read_csv(
        f'{INPUT}/test.csv', sep=',', header=0, index_col=0))
    test = test.rename(columns={
        'doughnuts_comsumption': 'required_doughnuts_comsumption'})

    # NECESSARY PREPROCESSING (nan removal & encoding into strings)
    test = basic_preprocessing(test)
    test.drop(labels=BINARY_NAN_FEATURES, axis=1, inplace=True)

    # # ONE-HOT ENCODINGS
    # logger.debug(f'\tApplying the encoders to one-hot encode the short categorical features')
    # for _f in BINARY_NAN_FEATURES:
    #     _encoded_arr: np.ndarray = onehot_encoders[_f].transform(test[_f].values.reshape(-1, 1)).toarray()
    #     _encoded_df = pd.DataFrame(
    #         _encoded_arr, columns=[f"{_f}_{_i + 1}" for _i in range(_encoded_arr.shape[1])], index=test.index)
    #     test = pd.concat([test.drop(labels=_f, axis=1), _encoded_df.astype(np.uint8)], axis=1)

    # TEXT EMBEDDINGS
    x_test = embed_text_features_separately(test, columns=EMBEDDED_FEATURES + ONEHOT_FEATURES, train=False)

    # x_test = pca.transform(x_test.toarray())

    # LAST CHECKS
    # perform_sanity_checks(test)
    # test = __reduce_bow_df_mem_usage(test)

    # TEST DATA SERIALIZATION
    if _save_data:
        logger.debug("\tSerializing the test data")
        with open(f'{OUTPUT}/test.npz', 'wb') as test_outfile:
            np.savez(test_outfile, x=x_test)

    return x_train, y_train, x_test


######################################################################
# TEXT EMBEDDINGS
######################################################################

def embed_text_features(df: pd.DataFrame, columns: List[str], train: bool) -> np.ndarray:
    global vectorizers
    logger.debug(f"\tApplying a TF-IDF vectorizer to embed the text of the categorical features")

    logger.debug(f"\t\tCleaning the merged text")
    text: pd.Series = clean_text_feature(df[columns].apply('. '.join, axis=1))

    if train:
        logger.debug(f"\t\tTraining the vectorizer to the merged text")
        vectorizers['all'] = TfidfVectorizer()
        # __check_vectorizer_not_trained(vectorizers['all'])
        text_vectors = vectorizers['all'].fit_transform(text)
        # __check_vectorizer_is_trained(vectorizers['all'])
    else:
        logger.debug(f"\t\tApplying the trained vectorizer to the merged text")
        # __check_vectorizer_is_trained(vectorizers['all'])
        text_vectors = vectorizers['all'].transform(text)

    _train_arr: np.ndarray = df.drop(labels=columns, axis=1)
    _train_arr = hstack((_train_arr, text_vectors))
    return _train_arr


def embed_text_features_separately(df: pd.DataFrame, columns: List[str], train: bool):
    global vectorizers
    logger.debug(f"\tApplying a TF-IDF vectorizer to embed the text of the categorical features")
    sparse_matrices: list = []
    for col in columns:
        text: pd.Series = clean_text_feature(df[col])
        if train:
            logger.debug(f"\t\tTraining the vectorizer to the column: {col}")
            vectorizers[col] = TfidfVectorizer()
            # __check_vectorizer_not_trained(vectorizers[col])
            text_vectors = vectorizers[col].fit_transform(text)
            # __check_vectorizer_is_trained(vectorizers[col])
        else:
            logger.debug(f"\t\tApplying the trained vectorizer to the column: {col}")
            # __check_vectorizer_is_trained(vectorizers[col])
            text_vectors = vectorizers[col].transform(text)
        sparse_matrices.append(text_vectors)

    sparse_final_matrix = sparse_matrices[0]
    for _s in sparse_matrices[1:]:
        sparse_final_matrix = hstack((sparse_final_matrix, _s))

    _feature_arr: np.ndarray = df.drop(labels=columns, axis=1)
    _feature_arr = hstack((_feature_arr, sparse_final_matrix))
    return _feature_arr


def __check_vectorizer_not_trained(vectorizer: TfidfVectorizer) -> None:
    from sklearn.utils.validation import check_is_fitted
    from sklearn.exceptions import NotFittedError
    try:
        check_is_fitted(vectorizer, '_tfidf', msg='The tfidf vector is not fitted')
    except NotFittedError:
        print("Indeed the vectorizer is not trained")
        return
    raise Exception("The vectorizer is already fitted!")


def __check_vectorizer_is_trained(vectorizer: TfidfVectorizer) -> None:
    from sklearn.utils.validation import check_is_fitted
    from sklearn.exceptions import NotFittedError
    try:
        check_is_fitted(vectorizer, '_tfidf', msg='The tfidf vector is not fitted')
        print("Indeed the vectorizer is trained")
    except NotFittedError as error:
        raise NotFittedError("The vectorizer is not trained!") from error


def clean_text_feature(df: pd.Series) -> pd.Series:
    # Lowercase all text
    df = df.str.lower()
    # Remove numbers
    df = df.str.replace(r'\d+', '', regex=True)
    # Remove punctuation
    df = df.str.replace(r'[^\w\s]', '', regex=True)
    # Remove extra spaces
    df = df.str.strip()
    # Remove Stopwords
    df = df.apply(lambda x: " ".join(x for x in x.split() if x not in stop))
    return df


#######################################################################
# BASIC PREPROCESSING
#######################################################################

def basic_preprocessing(data: pd.DataFrame) -> pd.DataFrame:
    data = _location_extraction(data)
    data = _empty_strings_as_nan(data, BINARY_NAN_FEATURES + ONEHOT_FEATURES + EMBEDDED_FEATURES)
    data.drop(labels=[NUMERICAL_FEATURE, 'job_id'], axis=1)
    data = _binary_nan_encoding(data, BINARY_NAN_FEATURES)
    data = _encode_nan_as_strings(data)
    data[SYNTHETIC_FEATURE] = _add_nan_per_sample(data)

    return data


def _encode_nan_as_strings(data: pd.DataFrame) -> pd.DataFrame:
    logger.debug("\tEncoding the nans as '{col_name}_nan' strings.")
    for col in data.columns:
        new_category = "nan"
        if data[col].dtype == 'category':
            with catch_warnings():
                simplefilter(action='ignore')
                try:
                    data[col].cat.add_categories([new_category], inplace=True)
                except ValueError:
                    # it means the new_category is already in the dataset
                    # (it comes from the binary nan encoding, i.e. location)
                    pass
                data[col].fillna(new_category, inplace=True)
        else:
            data[col].fillna(new_category, inplace=True)

    return data


def _empty_strings_as_nan(data: pd.DataFrame, columns: List[str]) -> pd.DataFrame:
    """
    Encode the nans (and empty strings) of a categorical feature by
    assigning them a new category: "{feature_name}_nan".
    Parameters
    ----------
    data: pd.DataFrame
        Raw dataframe with nans (containing both features & labels)
    columns: List[str]
        List of columns in which act
    Returns
    -------
    pd.DataFrame
        Dataframe with nans set as new categorical category for each feature
    """
    logger.debug("\tEncoding the empty strings as np.nan.")
    data[columns] = data[columns].copy().where(data[columns] != '', other=np.nan)

    return data


def _binary_nan_encoding(data: pd.DataFrame, columns: List[str]) -> pd.DataFrame:
    logger.debug('\tBinary encoding into 2 categories: nan & not nan')
    for col in columns:
        new_categories = [f"{str(col).replace(' ', '_')}_notnan",
                          f"{str(col).replace(' ', '_')}_nan"]
        with catch_warnings():
            simplefilter(action='ignore')
            data[col] = data[col].astype('category')
            data[col].cat.add_categories(new_categories, inplace=True)
            data[col] = data[col].where(
                pd.isna(data[col]), other=new_categories[0])
            data[col] = data[col].where(
                ~pd.isna(data[col]), other=new_categories[-1])

    return data


def _location_extraction(data: pd.DataFrame) -> pd.DataFrame:
    """
    Several tasks regarding feature extraction and postprocessing performed:
    - Location is split in 'country', 'state' & 'city'
    - Replace the categories consisting in empty strings '' (for the
    categorical variables) by np.nan.

    Parameters
    ----------
    data
    Returns
    -------
    """
    logger.debug("\tExtracting features manually")
    # we separate location into the following 2 features
    cols: list = ['country', 'state', 'city']
    data[cols] = data['location'].str.split(',', expand=True).loc[:, :2]
    data.drop(columns=['location'], axis=1, inplace=True)

    return data


def _add_nan_per_sample(features: pd.DataFrame) -> pd.Series:
    nan_per_sample = np.count_nonzero(pd.isna(features), axis=1)
    return pd.Series(nan_per_sample, name=SYNTHETIC_FEATURE)


###################################################################################################################
# AUXILIARY
##################################################################################################################


def perform_sanity_checks(dataframe: pd.DataFrame) -> None:
    nans: int = np.count_nonzero(pd.isna(dataframe))
    _feature_sample_ratio: float = len(dataframe.columns) / len(dataframe)
    if _feature_sample_ratio > 0.1:
        logger.warning(f"\t\tThe feature / sample ratio is above 0.1: {_feature_sample_ratio}!")
    else:
        logger.debug(f"\t\tCheck passed: feature / sample ratio below 0.1: {_feature_sample_ratio}")

    if nans > 0:
        logger.warning(f"\t\tThere are still NaN in the data: {nans}. "
                       f"Consider handling them before training.")
    else:
        logger.debug("\t\tCheck passed: there is no NaN in the data")


def __reduce_mem_usage(dataframe: pd.DataFrame,
                       _print_details: bool = False) -> pd.DataFrame:
    start_mem_usg = dataframe.memory_usage().sum() / 1024 ** 2
    logger.debug(f"\tMemory usage of initial dataframe: {start_mem_usg} MB")
    na_list = []  # Keeps track of columns that have missing values filled in.
    for col in dataframe.columns:
        if dataframe[col].dtype != object:  # Exclude strings
            # Print current column type
            _previous_dtype = dataframe[col].dtype

            # make variables for Int, max and min
            is_int = False
            mx = dataframe[col].max()
            mn = dataframe[col].min()

            # Integer does not support NA, therefore, NA needs to be filled
            if not np.isfinite(dataframe[col]).all():
                na_list.append(col)
                dataframe[col].fillna(mn - 1, inplace=True)

            # test if column can be converted to an integer
            as_int = dataframe[col].fillna(0).astype(np.int64)
            result = (dataframe[col] - as_int)
            result = result.sum()
            if -0.01 < result < 0.01:
                is_int = True

            # Make Integer/unsigned Integer datatypes
            if is_int:
                if mn >= 0:
                    if mx < 255:
                        dataframe[col] = dataframe[col].astype(np.uint8)
                    elif mx < 65535:
                        dataframe[col] = dataframe[col].astype(np.uint16)
                    elif mx < 4294967295:
                        dataframe[col] = dataframe[col].astype(np.uint32)
                    else:
                        dataframe[col] = dataframe[col].astype(np.uint64)
                else:
                    if mn > np.iinfo(np.int8).min and mx < np.iinfo(
                            np.int8).max:
                        dataframe[col] = dataframe[col].astype(np.int8)
                    elif mn > np.iinfo(np.int16).min and mx < np.iinfo(
                            np.int16).max:
                        dataframe[col] = dataframe[col].astype(np.int16)
                    elif mn > np.iinfo(np.int32).min and mx < np.iinfo(
                            np.int32).max:
                        dataframe[col] = dataframe[col].astype(np.int32)
                    elif mn > np.iinfo(np.int64).min and mx < np.iinfo(
                            np.int64).max:
                        dataframe[col] = dataframe[col].astype(np.int64)
            else:
                dataframe[col] = dataframe[col].astype(np.float32)

            if _print_details:
                logger.debug(f"Col: {col}. Dtype before: {_previous_dtype} "
                             f"---> dtype after: {dataframe[col].dtype}")

    mem_usg = dataframe.memory_usage().sum() / 1024 ** 2
    logger.debug(f"\tMemory usage AFTER reduction: {mem_usg} MB; "
                 f"i.e. {100 * mem_usg / start_mem_usg}% of the initial size")

    if len(na_list) > 0:
        logger.warning("The following columns present NaN and they have "
                       "been replaced by dataframe[col].min() -1:",
                       ", ".join(na_list))
    return dataframe


if __name__ == '__main__':
    process_dataframes(_save_data=True)  # _save_data=True)
