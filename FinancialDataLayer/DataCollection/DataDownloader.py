import pandas as pd
import yfinance as yf

from FinancialDataLayer.DataCollection.DatasetCollector import DatasetCollector


# TODO: add functionality to download minute data
class DataDownloader(DatasetCollector):
    """
    Provides methods for retrieving daily security data from Yahoo Finance API

    Attributes
    ----------

    Methods
    -------
    """

    def __init__(self, start_date: str, end_date: str, ticker_list: list, interval="1d", proxy=None):
        self.start_date = start_date
        self.end_date = end_date
        self.ticker_list = ticker_list
        self.interval = interval
        self.proxy = proxy

    def collect(self):
        df = self.download_from_yahoo()
        return df

    def download_from_yahoo(self) -> pd.DataFrame:
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
        for tic in self.ticker_list:
            temp_df = yf.download(tic, start=self.start_date, end=self.end_date, interval=self.interval,
                                  proxy=self.proxy)
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
                "tic"
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

        # if self.interval in ["1m", "2m", "5m", "15m", "30m”, “60m", "90m", "1h"]:
        #     data_df["date"] = data_df.date.apply(lambda x: x.strftime("%Y-%m-%d-%HH-%MM"))
        # else:
        #     data_df["date"] = data_df.date.apply(lambda x: x.strftime("%Y-%m-%d"))
        # drop missing data
        data_df = data_df.dropna()
        data_df = data_df.reset_index(drop=True)
        print("Shape of DataFrame: ", data_df.shape)
        # print("Display DataFrame: ", data_df.head())

        data_df = data_df.sort_values(
            by=["date", "tic"]).reset_index(drop=True)

        return data_df
