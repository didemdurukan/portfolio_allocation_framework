import yaml
from AgentLayer.RLAgents.DDPG import DDPG
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

    tickers = user_params["TICKERS"]
    env_kwargs = user_params["ENV_PARAMS"]
    train_params = user_params["TRAIN_PARAMS"]
    policy_params = user_params["POLICY_PARAMS"]
    test_params = user_params["TEST_PARAMS"]

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

    #CREATE DDPG AGENT
    ddpg = DDPG(env = env_train, **policy_params["DDPG_PARAMS"])

    #TRAIN DDPG AGENT
    ddpg.train_model(**train_params["DDPG_PARAMS"])

    #CREATE TEST ENV
    env_test = PortfolioEnv(df=df_processed, **env_kwargs) 

    #TEST DDPG AGENT
    ddpg.predict(environment = env_test, **test_params["DDPG_PARAMS"])

    #SAVE AGENT
    ddpg.save_model("AgentLayer/RLAgents/ddpg_model")

    #LOAD AGENT 
    loaded_ddpg_model = ddpg.load_model("AgentLayer/RLAgents/ddpg_model")

    print(loaded_ddpg_model)