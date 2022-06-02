from pyfolio import timeseries
import pyfolio
import pandas as pd
from FinancialDataLayer.DataCollection.DataDownloader import DataDownloader
import warnings
import yaml

warnings.simplefilter(action='ignore', category=FutureWarning)
#config = yaml.safe_load(open("../user_params.yaml"))
config = yaml.safe_load(open("user_params.yaml"))#bende boyleyken calisiyor 

class PortfolioEvaluator:

    def __init__(self, portfolio_df):
        self.portfolio_df = portfolio_df
        self.stats = None

    def backtest_stats(self, value_col_name="account_value"):
        dr_test = self._get_daily_return(self.portfolio_df, value_col_name=value_col_name)
        self.stats = timeseries.perf_stats(
            returns=dr_test,
            factor_returns=None,  # bechmark_returns
            positions=None,
            transactions=None,
            turnover_denom="AGB"
        )
        perf_stats_all = self.stats
        return perf_stats_all

    def backtest_plot(self,
                      baseline_start=config["TRADE_START_DATE"],
                      baseline_end=config["TRADE_END_DATE"],
                      baseline_ticker="^DJI",
                      value_col_name="account_value",
                      ):
        df = self.portfolio_df.copy()
        df["date"] = pd.to_datetime(df["date"])
        test_returns = self._get_daily_return(df, value_col_name=value_col_name)

        baseline_df = self._get_baseline(
            ticker=baseline_ticker, start=baseline_start, end=baseline_end
        )

        baseline_df["date"] = pd.to_datetime(baseline_df["date"], format="%Y-%m-%d")
        baseline_df = pd.merge(df[["date"]], baseline_df, how="left", on="date")
        baseline_df = baseline_df.fillna(method="ffill").fillna(method="bfill")
        baseline_returns = self._get_daily_return(baseline_df, value_col_name="close")

        with pyfolio.plotting.plotting_context(font_scale=1.1):
            pyfolio.create_full_tear_sheet(
                returns=test_returns, benchmark_rets=baseline_returns, set_context=False
            )

    @staticmethod
    def _get_daily_return(portfolio_df, value_col_name="account_value"):
        df = portfolio_df.copy()
        df["daily_return"] = df[value_col_name].pct_change(1)
        df["date"] = pd.to_datetime(df["date"])
        df.set_index("date", inplace=True, drop=True)
        df.index = df.index.tz_localize("UTC")
        return pd.Series(df["daily_return"], index=df.index, dtype='float64')

    @staticmethod
    def _get_baseline(ticker, start, end):
        return DataDownloader(start_date=start, end_date=end, ticker_list=[ticker]).download_from_yahoo()

# Class Attributes:
# Model Returns(pandas series)
# Portfolio Values(pandasseries / pyfoliotimeseries)
# Model actions(list)
#
# def __init__():
#     Initiates the backtest class object,
# gets the returns of the prediction and converts them to pandas series object that is in same format with pyfolio time series object
# Input:
# model returns(pandas data frame),
# portfolio values(pandas data frame),
# model_actions(list)
# Output: —-
#
# def perf_stats():
#     Calculates the performance metrics given as parameter
#
# Input: performance metrics(list)
# Output: performance values(pandas data frame)
#
# def perf_plot():
#     Plot the backtest results with given parameters
#
# Input: start
# time(string), end
# time(string), plot
# parameters(dict)
# Output: plot
# of
# the
# performance
# values
