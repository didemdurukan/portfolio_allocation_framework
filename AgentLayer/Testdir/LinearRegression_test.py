from statistics import median
from AgentLayer.ConventionalAgents.LinearRegression import LinearRegressionAgent
from AgentLayer.DataSplitter.TimeSeriesSplitter import TimeSeriesSplitter
from FinancialDataLayer.DataCollection.DataDownloader import DataDownloader
from FinancialDataLayer.DataProcessing.DefaultFeatureEngineer import DefaultFeatureEngineer
import yaml
import numpy as np
from AgentLayer.metrics import *

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
    tech_indicator_list = env_kwargs["tech_indicator_list"]

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
    print("Raw Data: ", downloaded_df.head())

    # PREPROCESS DATA
    print("\nTest 4: Feature engineer.........")
    data_processor = DefaultFeatureEngineer(use_default=False,
                                            tech_indicator_list=tech_indicator_list,
                                            use_vix=True,
                                            use_turbulence=True,
                                            use_covar=True)
    # included technical indicators as features
    df_processed = data_processor.extend_data(downloaded_df)

    # split data to train and test
    splitter = TimeSeriesSplitter()
    train = splitter.get_split_data(df_processed, '2009-01-01', '2020-06-30')
    trade = splitter.get_split_data(df_processed, '2020-07-01', '2021-09-02')

    # Get unique tic and trade
    unique_tic = trade.tic.unique()
    unique_trade_date = trade.date.unique()

    x_train, y_train = data_processor.prepare_ml_data(train)

    # Create Linear Regression model and train it
    lr = LinearRegressionAgent()
    trained_lr = lr.train_model(x_train, y_train)

    # Predict
    portfolio, portfolio_cumprod, meta_coefficient = lr.predict(
        trained_lr, 1000000, df_processed, unique_trade_date, tech_indicator_list)

    print("portfolio: \n", portfolio)
    print("portfolio_cumprod: \n", portfolio_cumprod)
    print("Meta Coefficient: \n", meta_coefficient)

    '''
    y_pred = lr.predict_test(trained_lr, test_x)
    print("y pred: ", y_pred)

    # Max Error
    maxError = max_error(test_y, y_pred)
    print("max error: ", maxError)

    # Mean Absolute Error
    mae = mean_absolute_error(test_y, y_pred)
    print("mae error: ", mae)

    # Mean Squared Error
    mse = mean_squared_error(test_y, y_pred)
    print("mse: ", mse)

    # Median Absolute Error
    median_abs_err = median_absolute_error(test_y, y_pred)
    print("median_abs_err: ", median_abs_err)

    # r^2 Score
    r2_scr = r2_score(test_y, y_pred)
    print("r2_scr: ", r2_scr)

    # Explained Varience Score
    evs = explained_variance_score(test_y, y_pred)
    print("Explained Varience Score: ", evs)

    # Mean Tweedie Deviance
    mtd = mean_tweedie_deviance(test_y, y_pred)
    print("Mean tweedie deviance: ", mtd)

    # Mean Poisson Deviance
    mpd = mean_poisson_deviance(test_y, y_pred)
    print("Mean poisson deviance: ", mpd)

    # Mean Gamma Deviance
    mgd = mean_gamma_deviance(test_y, y_pred)
    print("Mean gamma devaince: ", mgd)

    '''
