# Test File
from customDataset import ImportCustomDataset
from downloadDataset import DownloadDataset
import pandas as pd

if __name__ == '__main__':

    # df = ImportCustomDataset("examplejson.json").createDataset()
    # print(df)

    tickers = ['AAPL', 'DOW']  # ticker types defined
    df = DownloadDataset(start_date='2009-01-01',
                         end_date='2021-10-31',
                         ticker_list=tickers).createDataset()  # get data from Yahoo Downloader
    print(df.head())