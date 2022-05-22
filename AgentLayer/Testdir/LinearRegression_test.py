from statistics import median
from AgentLayer.ConventionalAgents.LinearRegression import LinearRegressionModel
from FinancialDataLayer.DataCollection.DataDownloader import DataDownloader
from FinancialDataLayer.DataProcessing.DefaultFeatureEngineer import DefaultFeatureEngineer
import yaml
import numpy as np
from AgentLayer.metrics import *

if __name__ == '__main__':
    print("main")
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

    '''
    x = np.random.normal(3, 1, 100)
    y = np.random.normal(150, 40, 100) / x

    train_x = x[:80]
    train_y = y[:80]

    train_x = train_x.reshape(-1, 1)
    train_y = train_y.reshape(-1, 1)

    test_x = x[80:]
    test_y = y[80:]

    test_x = test_x.reshape(-1, 1)
    test_y = test_y.reshape(-1, 1)


    '''

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
    df_processed = data_processor.extend_data(downloaded_df)  # included technical indicators as features
    # use covar = True
    print("Preprocessed Data: ", df_processed.head())

    x_train, y_train = data_processor.prepare_ml_data(df_processed)
    print("ml x:", x_train.head())
    print("ml Y: ", y_train.head())

    # SPLIT TO X AND Y

    # TODO: add return value to the imported data

    # lr = LinearRegressionModel()
    # df_processed_X, df_processed_Y = lr.split_x_y(
    #    df_processed, tech_indicator_list, tickers)
    # print(df_processed_Y.head())

    '''
    lr = LinearRegressionModel()
    trained_lr = lr.train_model(train_x, train_y)
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
