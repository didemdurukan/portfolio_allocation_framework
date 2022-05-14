from AgentLayer.ConventionalAgents.LinearRegression import LinearRegression
from FinancialDataLayer.DataCollection.DataDownloader import DataDownloader
from FinancialDataLayer.DataProcessing.DefaultFeatureEngineer import DefaultFeatureEngineer
from datasplitter import ExpandingWindowSplitter, BlockingTimeSeriesSplitter
import yaml

if __name__ == "main":

    # IMPORT .yaml FILE
    # Gather user parameters
    with open("user_params.yaml", "r") as stream:
        try:
            user_params = yaml.safe_load(stream)
        except yaml.YAMLError as exc:
            print(exc)

    tickers = user_params["tickers"]
    env_kwargs = user_params["env_params"]

    # FETCH DATA
    print("\nTest 3: Downloading from Yahoo.........")
    downloaded_df = DataDownloader(start_date='2009-01-01',
                                   end_date='2021-10-31',
                                   ticker_list=tickers).download_from_yahoo()

    print(downloaded_df.head())

    # PREPROCESS DATA
    print("\nTest 4: Feature engineer.........")

    df_processed = DefaultFeatureEngineer(use_default=False,
                                          tech_indicator_list=env_kwargs["tech_indicator_list"],
                                          use_vix=True,
                                          use_turbulence=True,
                                          use_covar=False).extend_data(downloaded_df)  # included technical indicators as features

    print(df_processed.head())

    # TRAIN-TEST SPLIT
