# Read YAML file
import yaml
import config
from AgentLayer.agents import A2C
from AgentLayer.finenvironment import PortfolioEnv
from FinancialEnvLayer.datacollector import DataDownloader
from FinancialEnvLayer.dataprocessor import FeatureEngineer
import tensorflow as tf

# Gather user parameters
with open("../user_params.yaml", "r") as stream:
    try:
        user_params = yaml.safe_load(stream)
    except yaml.YAMLError as exc:
        print(exc)

tickers = user_params["tickers"]
env_kwargs = user_params["env_params"]
# Custom MLP policy and value networks each with two layers of size [32 ,32] each with tanh activation function
policy_kwargs = user_params["policy_params"]
print(policy_kwargs)
agent_kwargs = user_params["agent_params"]
train_kwargs = user_params["train_params"]

# Download data from Yahoo
print("\nTest 1: Downloading from Yahoo.........")
downloaded_df = DataDownloader.download_data(start_date='2009-01-01',
                                             end_date='2021-10-31',
                                             ticker_list=tickers)

# Process data: add features
print("\nTest 2:Feature engineering.........")
df_processed = FeatureEngineer.add_features(df=downloaded_df,
                                            use_default=True,
                                            use_covar=True,
                                            use_vix=True,
                                            use_turbulence=True,
                                            user_defined_feature=True)  # included technical indicators as features

print("\nTest 3: Environment creation.........")
# Create environment
env_train = PortfolioEnv(df=df_processed,
                         **env_kwargs)  # train parametresi training data olucak (dataframe) daha koyulmadi

env_trade = PortfolioEnv(df=df_processed, **env_kwargs)  # trade parametresi test data olucak (dataframe) daha koyulmadi

print("\nTest 4: Agent creation.........")
# object creation
a2c = A2C(env=env_train, **agent_kwargs)
# # training
# a2c.train_model(**train_kwargs)

# # Future work
# # predicting
# a2c.predict(test_params)
