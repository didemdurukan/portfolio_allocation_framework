# Test File
import config
from datacollector import CustomDatasetImporter
from datacollector import DataDownloader
from dataprocessor import FeatureEngineer
import yaml

import pandas as pd

if __name__ == '__main__':

    print("\nTest 1: Loading from json............")
    json_df = CustomDatasetImporter.from_file("examplejson.json")
    print(json_df.head())

    print("\nTest 2: Loading from custom df.........")
    df_to_load = pd.DataFrame(data=[5, 4, 3])
    custom_df = CustomDatasetImporter.from_df(df_to_load)
    print(custom_df.head())

    # Read YAML file
    with open("../user_params.yaml", "r") as stream:
        try:
            user_params = yaml.safe_load(stream)
        except yaml.YAMLError as exc:
            print(exc)

    tickers = user_params["tickers"]

    print("\nTest 3: Downloading from Yahoo.........")
    downloaded_df = DataDownloader.download_data(start_date='2009-01-01',
                                                 end_date='2021-10-31',
                                                 ticker_list=tickers)
    print(downloaded_df.head())

    print("\nTest 4: Feature engineer.........")
    df_processed = FeatureEngineer.add_features(df=downloaded_df,
                                                use_default=True,
                                                use_covar=True,
                                                use_vix=True,
                                                use_turbulence=True,
                                                user_defined_feature=True)  # included technical indicators as features
    print("\nAdded features....")

    print(df_processed.head(1))
