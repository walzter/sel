import os
from ..file_reader import (
    get_feature_name_small,
    get_feature_name_medium,
    get_feature_name_large,
)
from ..cleaners import replace_column_values
import pandas as pd
import numpy as np
import gc


class Cleaner:
    def __init__(self, file_path: str, size: str, out_dir: str, verbosity: bool):
        self.file_path = file_path
        self.size = size
        self.out_dir = out_dir
        self.verb = verbosity

    def _make_dataframe(
        self,
    ) -> pd.DataFrame:
        """
        Function which takes the data_path and the labels_path and returns a dataframe
        """
        gc.collect()
        ## check if the out dir exists
        if not os.path.isdir(os.path.join(os.getcwd(), self.out_dir)):
            os.mkdir(os.path.join(os.getcwd(), self.out_dir))
        ## get the data
        data = list(filter(lambda x: x.endswith(".data"), self.file_path))[0]

        ## get the labels
        labels = list(filter(lambda x: x.endswith(".names"), self.file_path))[0]

        ## read the data as a csv
        df = pd.read_csv(data, header=None)
        del data
        ## get the column names
        if self.size == "small":
            col_names_small = get_feature_name_small(labels, verbosity=self.verb)
            df.columns = col_names_small[f"{self.size}"]["column_names"]
            ## clean
            _df = df.copy().replace("?", np.nan)
            _df.dropna(axis=0, inplace=True)
            del df
            _df.to_csv(os.path.join(self.out_dir, "small_expanded.csv"))
            return _df
        elif self.size == "medium":
            col_names_medium = get_feature_name_medium(labels, verbosity=self.verb)
            df.columns = col_names_medium[f"{self.size}"]["column_names"]
            ## clean
            _df = df.copy().replace("?", np.nan)
            _df.dropna(axis=0, inplace=True)
            del df
            _df.to_csv(os.path.join(self.out_dir, "medium_expanded.csv"))
            return _df
        elif self.size == "large":
            col_names_large, main_dict = get_feature_name_large(
                labels, verbosity=self.verb
            )
            ## change the column names
            df.columns = col_names_large["large"]["column_names"]
            _df = df.copy().replace("?", np.nan)
            _df.dropna(axis=0, inplace=True)
            ## now replace the values
            _df = replace_column_values(df, main_dict)
            _df.to_csv(os.path.join(self.out_dir, "large_expanded.csv"))
            return _df
