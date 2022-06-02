import yaml
from AgentLayer.RLAgents.PPO import PPO
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

    df_processed = DefaultFeatureEngineer(  use_default= False,
                                            tech_indicator_list= env_kwargs["tech_indicator_list"],
                                            use_vix=True,
                                            use_turbulence=True,
                                            use_covar=True).extend_data(downloaded_df)  # included technical indicators as features

    print(df_processed.head())

    #CREATE TRAIN ENV
    env = PortfolioEnv(df=df_processed, **env_kwargs) 
    env_train, _ = env.get_env()

    #CREATE PPO AGENT
    ppo = PPO(env = env_train, **policy_params["PPO_PARAMS"])

    #TRAIN PPO AGENT
    ppo.train_model(**train_params["PPO_PARAMS"])

    #CREATE TEST ENV
    env_test = PortfolioEnv(df=df_processed, **env_kwargs) 

    #TEST PPO AGENT
    ppo.predict(environment = env_test, **test_params["PPO_PARAMS"])

    #SAVE AGENT
    ppo.save_model("AgentLayer/RLAgents/ppo_model")

    #LOAD AGENT 
    loaded_ppo_model = ppo.load_model("AgentLayer/RLAgents/ppo_model")

    print(loaded_ppo_model)