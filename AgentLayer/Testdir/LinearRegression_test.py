from AgentLayer.ConventionalAgents.LinearRegression import LinearRegressionModel
#from FinancialDataLayer.DataCollection.DataDownloader import DataDownloader
#from FinancialDataLayer.DataProcessing.DefaultFeatureEngineer import DefaultFeatureEngineer
#from datasplitter import ExpandingWindowSplitter, BlockingTimeSeriesSplitter
import yaml
import numpy as np

if __name__ == '__main__':
    print("main")
    # IMPORT .yaml FILE
    # Gather user parameters
    with open("user_params.yaml", "r") as stream:
        try:
            user_params = yaml.safe_load(stream)
        except yaml.YAMLError as exc:
            print(exc)

#    tickers = user_params["tickers"]
#    env_kwargs = user_params["env_params"]

    x = np.random.normal(3, 1, 100)
    y = np.random.normal(150, 40, 100) / x

    train_x = x[:80]
    train_y = y[:80]

    train_x = train_x.reshape(-1, 1)
    train_y = train_y.reshape(-1, 1)

    test_x = x[80:]
    test_y = y[80:]

    lr = LinearRegressionModel()
    trained_lr = lr.train_model(train_x, train_y)
    lr.save_model(trained_lr, "test")
    my_lr_model = lr.load_model("test")
