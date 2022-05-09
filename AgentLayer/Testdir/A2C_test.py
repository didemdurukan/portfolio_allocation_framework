
from AgentLayer.RLAgents.A2C import A2C
from AgentLayer.Environment.PortfolioEnv import PortfolioEnv
from FinancialDataLayer.DataCollection.DataDownloader import DataDownloader
from FinancialDataLayer.DataProcessing.DefaultFeatureEngineer import DefaultFeatureEngineer
import yaml

#IMPORT .yaml FILE
#Gather user parameters
with open("../user_params.yaml", "r") as stream:
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

#CREATE A2C AGENT
a2c = A2C(env = env_train, **policy_params["A2C_PARAMS"])

#TRAIN A2C AGENT
a2c.train_model(**train_params["A2C_PARAMS"])

#CREATE TEST ENV
env_test = PortfolioEnv(df=df_processed, **env_kwargs) 

#TEST A2C AGENT
a2c.predict(environment = env_test, **test_params["A2C_PARAMS"])

#SAVE AGENT
a2c.save_model("RLAgentS/a2c_model")

#LOAD AGENT 
loaded_a2c_model = a2c.load_model("RLAgents/a2c_model")

print(loaded_a2c_model)