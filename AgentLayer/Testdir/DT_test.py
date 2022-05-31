import yaml
from AgentLayer.ConventionalAgents.DTAgent import DTAgent
from FinancialDataLayer.DataCollection.DataDownloader import DataDownloader
from FinancialDataLayer.DataProcessing.DefaultFeatureEngineer import DefaultFeatureEngineer
from AgentLayer.DataSplitter.TimeSeriesSplitter import TimeSeriesSplitter

if __name__ == '__main__':

    # IMPORT .yaml FILE
    # Gather user parameters
    with open("user_params.yaml", "r") as stream:
        try:
            user_params = yaml.safe_load(stream)
        except yaml.YAMLError as exc:
            print(exc)

    tickers = user_params["tickers"]
    env_kwargs = user_params["env_params"]
    train_params = user_params["train_params"]
    policy_params = user_params["policy_params"]
    test_params = user_params["test_params"]

    # FETCH DATA
    print("\nTest 3: Downloading from Yahoo.........")
    downloaded_df = DataDownloader(start_date='2009-01-01',
                                   end_date='2021-10-31',
                                   ticker_list=tickers).download_from_yahoo()
    """
    downloaded_df = DataDownloader.download_data(start_date='2009-01-01',
                                                    end_date='2021-10-31',
                                                ticker_list=tickers)
    """
    print(downloaded_df.head())

    # PREPROCESS DATA
    print("\nTest 4: Feature engineer.........")
    data_processor = DefaultFeatureEngineer(use_default=False,
                                            tech_indicator_list=env_kwargs["tech_indicator_list"],
                                            use_vix=True,
                                            use_turbulence=True,
                                            use_covar=True)
    # included technical indicators as features
    df_processed = data_processor.extend_data(downloaded_df)

    # split data to train and test
    splitter = TimeSeriesSplitter()
    train = splitter.get_split_data(df_processed, '2009-01-01', '2020-06-30')
    trade = splitter.get_split_data(df_processed, '2020-07-01', '2021-09-02')

    #prepare conventional data
    train_x, train_y = data_processor.prepare_ml_data(train)

    # create agent
    dt = DTAgent(**policy_params["DT_PARAMS"])

    #train agent
    dt.train_model(train_x, train_y, **train_params["DT_PARAMS"])

    #predict 
    portfolio, portfolio_cumprod, meta_coefficient = dt.predict(trade, **test_params["DT_PARAMS"])
    print(portfolio)

    #save model
    dt.save_model("AgentLayer/ConventionalAgents/dt_model")

    #load model
    dt_loaded = dt.load_model("AgentLayer/ConventionalAgents/dt_model")