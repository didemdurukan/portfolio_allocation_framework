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

    x_train = [1, 2, 4.5, 6, 7]
    y_train = [0.3, 3.5, 2.6, 4.6, 8]

    lr = LinearRegression()
    trained_lr = lr.train_model(x_train, y_train)
    lr.save_model(trained_lr, "test")
