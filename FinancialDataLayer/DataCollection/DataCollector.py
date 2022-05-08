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


# These helper functions are defined here because these methods can have potential uses later on in other classes
# The reasoning explained below: Is the helper function only for this
# class? If it can help in other places, then it goes at the module level, if it is only for this class, then it goes
# in the class with either: 1) static method (needs no class data to do its job) 2) class method (needs some class
# data but no instance data to do its job
def load_from_df(df_to_load) -> pd.DataFrame:
    return df_to_load


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
