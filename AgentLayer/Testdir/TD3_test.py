import yaml
from AgentLayer.RLAgents.TD3 import TD3
from AgentLayer.Environment.PortfolioEnv import PortfolioEnv
from FinancialDataLayer.DataCollection.DataDownloader import DataDownloader
from FinancialDataLayer.DataProcessing.DefaultFeatureEngineer import DefaultFeatureEngineer



if __name__ == '__main__':

    #IMPORT .yaml FILE
    #Gather user parameters
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

    #FETCH DATA
    print("\nTest 3: Downloading from Yahoo.........")
    downloaded_df = DataDownloader(start_date='2009-01-01',
                                    end_date='2021-10-31',
                                    ticker_list= tickers).download_from_yahoo()
    """
    downloaded_df = DataDownloader.download_data(start_date='2009-01-01',
                                                    end_date='2021-10-31',
                                                ticker_list=tickers)
    """
    print(downloaded_df.head())


    #PREPROCESS DATA
    print("\nTest 4: Feature engineer.........")

    df_processed = DefaultFeatureEngineer( use_default= False,
                                        tech_indicator_list= env_kwargs["tech_indicator_list"],
                                        use_vix=True,
                                        use_turbulence=True,
                                        use_covar=True).extend_data(downloaded_df)  # included technical indicators as features

    print(df_processed.head())

    #CREATE TRAIN ENV
    env = PortfolioEnv(df=df_processed, **env_kwargs) 
    env_train, _ = env.get_env()

    #CREATE TD3 AGENT
    td3 = TD3(env = env_train, **policy_params["TD3_PARAMS"])

    #TRAIN TD3 AGENT
    td3.train_model(**train_params["TD3_PARAMS"])

    #CREATE TEST ENV
    env_test = PortfolioEnv(df=df_processed, **env_kwargs) 

    #TEST TD3 AGENT
    td3.predict(environment = env_test, **test_params["TD3_PARAMS"])

    #SAVE AGENT
    td3.save_model("AgentLayer/RLAgents/td3_model")

    #LOAD AGENT 
    loaded_td3_model = td3.load_model("AgentLayer/RLAgents/td3_model")

    print(loaded_td3_model)