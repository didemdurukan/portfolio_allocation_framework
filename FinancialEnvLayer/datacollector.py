"""Contains methods and classes to collect data from Yahoo Finance API or import custom dataset
"""

from abc import ABC, abstractmethod
import pandas as pd
import yfinance as yf
import json


class DatasetCollector(ABC):
    """"
    This abstract base class defines the base structure of data collectors
    """


class DataDownloader(DatasetCollector):
    """
    Provides methods for retrieving daily security data from Yahoo Finance API

    Attributes
    ----------

    Methods
    -------
    """

    def __init__(self):
        pass

    @classmethod
    def download_data(cls, start_date: str, end_date: str, ticker_list: list, proxy=None):
        df = cls.download_from_yahoo(start_date, end_date, ticker_list, proxy)
        return df

    @staticmethod
    def download_from_yahoo(start_date: str, end_date: str, ticker_list: list, proxy=None) -> pd.DataFrame:
        """Fetches data from Yahoo Finance API
        Parameters
        ----------

        Returns
        -------
        `pd.DataFrame`
            7 columns: A date, open, high, low, (adjusted) close, volume and tick symbol
            for the specified security ticker
        """
        # Download and save the data in a pandas DataFrame:
        data_df = pd.DataFrame()
        for tic in ticker_list:
            temp_df = yf.download(tic, start=start_date, end=end_date, proxy=proxy)
            temp_df["tic"] = tic
            data_df = pd.concat([data_df, temp_df])
        # reset the index, we want to use numbers as index instead of dates
        data_df = data_df.reset_index()
        try:
            # convert the column names to standardized names
            data_df.columns = [
                "date",
                "open",
                "high",
                "low",
                "close",
                "adjclose",
                "volume",
                "tic",
            ]
            # use adjusted close price instead of close price
            data_df["close"] = data_df["adjclose"]
            # drop the adjusted close price column
            data_df = data_df.drop(labels="adjclose", axis=1)
        except NotImplementedError:
            print("the features are not supported currently")
        # create day of the week column (monday = 0)
        data_df["day"] = data_df["date"].dt.dayofweek
        # convert date to standard string format, easy to filter
        data_df["date"] = data_df.date.apply(lambda x: x.strftime("%Y-%m-%d"))
        # drop missing data
        data_df = data_df.dropna()
        data_df = data_df.reset_index(drop=True)
        print("Shape of DataFrame: ", data_df.shape)
        # print("Display DataFrame: ", data_df.head())

        data_df = data_df.sort_values(by=["date", "tic"]).reset_index(drop=True)

        return data_df

    def select_equal_rows_stock(self, df):
        df_check = df.tic.value_counts()
        df_check = pd.DataFrame(df_check).reset_index()
        df_check.columns = ["tic", "counts"]
        mean_df = df_check.counts.mean()
        equal_list = list(df.tic.value_counts() >= mean_df)
        names = df.tic.value_counts().index
        select_stocks_list = list(names[equal_list])
        df = df[df.tic.isin(select_stocks_list)]
        return df


# User Imports his/her own dataset

class CustomDatasetImporter(DatasetCollector):

    def __init__(self):
        pass

    @staticmethod
    def from_df(df):
        df = CustomDatasetImporter.__load_from_df(df_to_load=df)
        return df

    @staticmethod
    def from_file(filename) -> pd.DataFrame:
        df = CustomDatasetImporter.__load_from_file(path=filename)
        return df

    @staticmethod
    def __load_from_df(df_to_load) -> pd.DataFrame:
        return df_to_load

    @staticmethod
    def __load_from_file(path) -> pd.DataFrame:
        csv = ".csv"
        excel = ".xlsx"
        json_str = ".json"

        try:
            if csv in path:
                data = pd.read_csv(path)
            elif excel in path:
                data = pd.read_excel(path)
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
