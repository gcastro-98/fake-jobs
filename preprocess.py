import pandas as pd
import numpy as np
import spacy
from logger import Logger
from sklearn.preprocessing import OneHotEncoder
# from sklearn.manifold import TSNE, SpectralEmbedding
from sklearn.decomposition import KernelPCA
from typing import Dict, List, Any
from warnings import catch_warnings, simplefilter
from static import INPUT, OUTPUT, EMBEDDED_FEATURES, BINARY_NAN_FEATURES, ONEHOT_FEATURES, \
    NUMERICAL_FEATURE, SYNTHETIC_FEATURE


nlp = spacy.load('en_core_web_sm')  # spacy.load('en_core_web_md')
logger = Logger()


#######################################################################
# MAIN
#######################################################################

def process_dataframes() -> None:
    ################################################################
    # TRAIN DATAFRAME
    ################################################################

    logger.info(f"Loading the input train dataset")
    train: pd.DataFrame = __reduce_mem_usage(pd.read_csv(
        f'{INPUT}/train.csv', sep=',', header=0, index_col=0))

    # NECESSARY PREPROCESSING (nan removal & encoding into strings)
    train = basic_preprocessing(train)

    # ONE-HOT ENCODINGS
    logger.debug(f'\tOne-hot encoding the features: {", ".join(ONEHOT_FEATURES)}')
    onehot_encoders: Dict[str, Any] = {_f: OneHotEncoder() for _f in ONEHOT_FEATURES}
    for _f in ONEHOT_FEATURES:
        _encoded_arr: np.ndarray = onehot_encoders[_f].fit_transform(train[_f].values.reshape(-1, 1)).toarray()
        _encoded_df = pd.DataFrame(
            _encoded_arr, columns=[f"{_f}_{_i + 1}" for _i in range(_encoded_arr.shape[1])], index=train.index)
        train = pd.concat([train.drop(labels=_f, axis=1), _encoded_df], axis=1)

    # TEXT EMBEDDINGS
    logger.debug(f'\tEncoding the features using a text embedding (spacy + rbf-KernelPCA)')
    manifold_mappers: Dict[str, Any] = {
        _f: KernelPCA(n_components=_n, kernel='rbf') for _f, _n in EMBEDDED_FEATURES.items()}
    for _f, _n in EMBEDDED_FEATURES.items():
        # we first encode the text into a vector for each sample of the feature's pd.Series
        logger.debug(f"\t\tEncoding {_f} using spaCy")
        _encoded_arr = text_encode_feature(train[_f])
        logger.debug(f"\t\tEmbedding {_f} in a {_n} space using a rbf-KernelPCA")
        _embedded_arr: np.ndarray = manifold_mappers[_f].fit_transform(_encoded_arr)
        _embedded_df = pd.DataFrame(_embedded_arr, columns=[f"{_f}_{_i + 1}" for _i in range(_n)], index=train.index)
        train = pd.concat([train.drop(labels=_f, axis=1), _embedded_df], axis=1)

    # LAST CHECKS
    _perform_sanity_checks(train)
    # DATAFRAME SERIALIZATION
    train.to_csv(f'{OUTPUT}/train.csv', index=True)

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

    # ONE-HOT ENCODINGS
    logger.debug(f'\tApplying the encoders to one-hot encode the short categorical features')
    for _f in ONEHOT_FEATURES:
        _encoded_arr: np.ndarray = onehot_encoders[_f].transform(test[_f].values.reshape(-1, 1)).toarray()
        _encoded_df = pd.DataFrame(
            _encoded_arr, columns=[f"{_f}_{_i + 1}" for _i in range(_encoded_arr.shape[1])], index=test.index)
        test = pd.concat([test.drop(labels=_f, axis=1), _encoded_df], axis=1)

    # TEXT EMBEDDINGS
    logger.debug(f'\tApplying the mappers to encode features using a text embedding (spacy + t-SNE)')
    for _f, _n in EMBEDDED_FEATURES.items():
        # we first encode the text into a vector for each sample of the feature's pd.Series
        logger.debug(f"\t\tEncoding {_f} using spaCy")
        _encoded_arr = text_encode_feature(test[_f])
        logger.debug(f"\t\tEmbedding {_f} in a {_n} space using a rbf-KernelPCA")
        _embedded_arr: np.ndarray = manifold_mappers[_f].transform(_encoded_arr)
        _embedded_df = pd.DataFrame(_embedded_arr, columns=[f"{_f}_{_i + 1}" for _i in range(_n)], index=test.index)
        test = pd.concat([test.drop(labels=_f, axis=1), _embedded_df], axis=1)

    # LAST CHECKS
    _perform_sanity_checks(test)
    # DATAFRAME SERIALIZATION
    test.to_csv(f'{OUTPUT}/test.csv', index=True)


######################################################################
# TEXT EMBEDDINGS
######################################################################

def text_encode_feature(data: pd.Series) -> np.ndarray:
    # working: process the text with spaCy and get the vector representation
    # vector = nlp(text).vector
    def __encode(text: str):
        return nlp(text).vector[np.newaxis, :]

    # thus, applying it to the Series
    text_samples: np.ndarray = data.astype(str).values
    encoded_matrix = __encode(text_samples[0])
    for _sample in text_samples[1:]:
        encoded_matrix = np.concatenate([encoded_matrix, __encode(_sample)], axis=0)
    return encoded_matrix


#######################################################################
# BASIC PREPROCESSING
#######################################################################

def basic_preprocessing(data: pd.DataFrame) -> pd.DataFrame:
    data = _location_extraction(data)
    data = _empty_strings_as_nan(data, BINARY_NAN_FEATURES + ONEHOT_FEATURES)
    data.drop(labels=[NUMERICAL_FEATURE, 'job_id'], axis=1, inplace=True)
    data = _encode_nan_as_strings(data)
    data[SYNTHETIC_FEATURE] = _add_nan_per_sample(data)
    data = _binary_nan_encoding(data, BINARY_NAN_FEATURES)
    return data


def _encode_nan_as_strings(data: pd.DataFrame) -> pd.DataFrame:
    logger.debug("\tEncoding the nans as '{col_name}_nan' strings.")
    for col in data.columns:
        if data[col].dtype == 'category':
            new_categories = [f"{str(col).replace(' ', '_')}_nan"]
            with catch_warnings():
                simplefilter(action='ignore')
                try:
                    data[col].cat.add_categories(new_categories, inplace=True)
                except ValueError:
                    # it means the new_category is already in the dataset
                    # (it comes from the binary nan encoding, i.e. location)
                    pass
                data[col].fillna(new_categories[0], inplace=True)

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


def _perform_sanity_checks(dataframe: pd.DataFrame) -> None:
    nans: int = np.count_nonzero(np.isnan(dataframe))

    if __feature_sample_ratio_is_high(dataframe):
        logger.warning("\t\tThe feature / sample ratio is above 0.1!")
    else:
        logger.debug("\t\tCheck passed: feature / sample ratio below 0.1")

    if nans > 0:
        logger.warning(f"\t\tThere are still NaN in the data: {nans}. "
                       f"Consider handling them before training.")
    else:
        logger.debug("\t\tCheck passed: there is no NaN in the data")


def __feature_sample_ratio_is_high(dataframe: pd.DataFrame) -> bool:
    return len(dataframe.columns) / len(dataframe) > 0.1


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
    process_dataframes()
