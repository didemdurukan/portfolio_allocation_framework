# Test File
import config
from FinancialDataLayer.DataCollection.CustomDatasetImporter import CustomDatasetImporter
from FinancialDataLayer.DataCollection.DataDownloader import DataDownloader
from FinancialDataLayer.DataProcessing.DefaultFeatureEngineer import DefaultFeatureEngineer
from FinancialDataLayer.DataProcessing.CustomFeatureEngineer import CustomFeatureEngineer
import yaml

import pandas as pd

if __name__ == '__main__':

    print("\nTest 1: Loading from json............")
    json_df = CustomDatasetImporter("examplejson.json").collect()
    print(json_df.head())

    print("\nTest 2: Loading from custom df.........")
    df_to_load = pd.DataFrame(data=[5, 4, 3])
    custom_df = CustomDatasetImporter(df=df_to_load).collect()
    print(custom_df.head())

    # Read YAML file
    with open("../user_params.yaml", "r") as stream:
        try:
            user_params = yaml.safe_load(stream)
        except yaml.YAMLError as exc:
            print(exc)

    tickers = user_params["tickers"]

    print("\nTest 3: Downloading from Yahoo.........")
    downloaded_df = DataDownloader(start_date='2009-04-01',
                                   end_date='2021-04-29',
                                   interval="1d",
                                   ticker_list=tickers).collect()
    print(downloaded_df.head())

    print("\nTest 4: Default Feature Engineer.........")
    df_processed = DefaultFeatureEngineer(use_default=True,
                                          use_covar=True,
                                          use_vix=True,  # VIX volatility index
                                          use_turbulence=True) \
        .extend_data(downloaded_df)  # included technical indicators as features

    print("\nAdded features....")
    print(df_processed.head())

    print("\nTest 5: Custom Feature Engineer....")
    df_processed = CustomFeatureEngineer(lag=2).extend_data(downloaded_df)
    print("\nAdded custom features")
    print(df_processed.head())


