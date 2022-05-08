import json
from abc import ABC, abstractmethod

import pandas as pd


class DatasetCollector(ABC):
    """"
    This abstract base class defines the base structure of data collectors
    """

    @abstractmethod
    def collect(self):
        pass

    # These helper functions are defined here as staticmethods these methods have uses later on in other classes that inherit DatasetCollector
    @staticmethod
    def load_from_df(df_to_load) -> pd.DataFrame:
        return df_to_load

    @staticmethod
    def load_from_file(path) -> pd.DataFrame:
        csv = ".csv"
        excel = ".xlsx"
        json_str = ".json"

        try:
            if csv in path:
                data = pd.read_csv(path)
            elif excel in path:
                data = pd.read_excel(path, sheet_name=0)
            elif json_str in path:
                with open(path) as json_file:
                    json_data = json.load(json_file)
                    data = pd.json_normalize(json_data)
            else:
                raise ValueError(
                    f"Unexpected path input. Please try with .csv, .xlsx, .json files"
                )
            return data
        except FileNotFoundError:
            print("File not found.")
