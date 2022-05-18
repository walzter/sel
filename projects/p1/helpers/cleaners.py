import pandas as pd

## helper functions to clean the dataframe
import gc


def replace_column_values(dataframe: pd.DataFrame, dictionary: dict):
    """
    Replaces the abbreviation of the values in the dataframe with the corresponding ones.
    e.g. feature 1: k, is replaced to feature 1: wood.

    PARAMS:
    -------
    dataftrame : dataframe which contains the columns to be replaced

    dictionary : dict with the mapping values

    RETURNS:
    --------
    dataframe : with replaced abbreviations.
    """
    ## iterate through the main_dict and replace the possible values with the actual values in the datagrame
    gc.collect()
    df = dataframe.copy(deep=True)
    _ = [
        df[feature].replace(replacements, inplace=True)
        for feature, replacements in dictionary.items()
    ]
    return df
