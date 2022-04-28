# Test File
from customDataset import ImportCustomDataset
from downloadDataset import DownloadDataset
import yaml

import pandas as pd

if __name__ == '__main__':

    # df = ImportCustomDataset("examplejson.json").createDataset()
    # print(df)

    # Read YAML file
    with open("download.yaml", "r") as stream:
        try:
            download_config = yaml.safe_load(stream)
        except yaml.YAMLError as exc:
            print(exc)

    tickers = download_config["tickers"]

    df = DownloadDataset(start_date='2009-01-01',
                         end_date='2021-10-31',
                         ticker_list=tickers)\
        .createDataset()  # get data from Yahoo Finance API Downloader

    print(df.head())


