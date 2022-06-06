from pyfolio import timeseries
import pyfolio
import pandas as pd
from FinancialDataLayer.DataCollection.DataDownloader import DataDownloader
import warnings
import yaml
import os
from utils import get_project_root

warnings.simplefilter(action='ignore', category=FutureWarning)

ROOT_DIR = get_project_root()
CONFIG_PATH = os.path.join(ROOT_DIR, 'user_params.yaml')

config = yaml.safe_load(open(CONFIG_PATH))


class PortfolioEvaluator:
    """Provides methods for portfolio evaluation.

        Attributes
        ----------
            -portfolio_dfs : pd.DataFrame
                portfolios to be evaluated.
            -agent_names : list
                agent names

        Methods
        -------
            backtest_stats()
                Calculates backtest statistics using _get_stats() helper function.
            _get_stats()
                gets the performance statistics.
            _create_backtest_plots()
                creates backtest plots.
            -backtest_plot()
                generates plots using _create_backtest_plots()
                helper function.
            -_get_daily_return()
                Gets daily return.
            -_get_baseline()
                downloads the data of the baseline from Yahoo API

    """

    def __init__(self, *portfolio_dfs, agent_names):
        self.portfolio_dfs = portfolio_dfs
        if agent_names is None:
            self.agent_names = ["Agent" + str(i)
                                for i in range(len(portfolio_dfs))]
        else:
            self.agent_names = agent_names
        self.stats_list = None

    def backtest_stats(self, value_col_name="account_value"):
        """Gets backtest statistics using _get_stats() helper function.

        Args:
            value_col_name (str, optional): Column name in the dataframe for calculating the statistics. Defaults to "account_value".

        Returns:
            pd.DataFrame: backtest statistics
        """
        perf_stats_list = []
        for portfolio in self.portfolio_dfs:
            perf_stats_list.append(self._get_stats(portfolio, value_col_name))
        self.stats_list = perf_stats_list
        return pd.DataFrame(perf_stats_list, index=self.agent_names)

    def _get_stats(self, portfolio_df, value_col_name="account_value"):
        """Calculates performance statistics.

        Args:
            portfolio_df (pd.DataFrame): portfolio data frame
            value_col_name (str, optional): Column name in the dataframe for calculating the statistics. Defaults to "account_value".

        Returns:
            pd.Series : Performance metrics.
        """
        dr_test = self._get_daily_return(
            portfolio_df, value_col_name=value_col_name)
        perf_stats = timeseries.perf_stats(
            returns=dr_test,
            factor_returns=None,
            positions=None,
            transactions=None,
            turnover_denom="AGB"
        )
        return perf_stats

    def _create_backtest_plot(self,
                              portfolio_df,
                              baseline_start=config["TRADE_START_DATE"],
                              baseline_end=config["TRADE_END_DATE"],
                              baseline_ticker="^DJI",
                              value_col_name="account_value",
                              output_col_name="Agent"
                              ):
        """Creates backtest plots

        Args:
            portfolio_df (pd.DataFrame): portfolio data frame
            baseline_start (str, optional): Start date for baseline. Defaults to config["TRADE_START_DATE"].
            baseline_end (str, optional): End date for baseline. Defaults to config["TRADE_END_DATE"].
            baseline_ticker (str, optional): Baseline ticker. Defaults to "^DJI".
            value_col_name (str, optional): Column name in the dataframe for calculating the statistics. Defaults to "account_value".
            output_col_name (str, optional): Output column name. Defaults to "Agent".
        """
        df = portfolio_df.copy()
        df["date"] = pd.to_datetime(df["date"])
        test_returns = self._get_daily_return(
            df, value_col_name=value_col_name, output_col_name=output_col_name)

        baseline_df = self._get_baseline(
            ticker=baseline_ticker, start=baseline_start, end=baseline_end
        )

        baseline_df["date"] = pd.to_datetime(
            baseline_df["date"], format="%Y-%m-%d")
        baseline_df = pd.merge(
            df[["date"]], baseline_df, how="left", on="date")
        baseline_df = baseline_df.fillna(method="ffill").fillna(method="bfill")
        baseline_returns = self._get_daily_return(
            baseline_df, value_col_name="close", output_col_name="Baseline")

        with pyfolio.plotting.plotting_context(font_scale=1.1):
            pyfolio.create_full_tear_sheet(
                returns=test_returns, benchmark_rets=baseline_returns
            )

    def backtest_plot(self,
                      baseline_start=config["TRADE_START_DATE"],
                      baseline_end=config["TRADE_END_DATE"],
                      baseline_ticker="^DJI",
                      value_col_name="account_value",
                      ):
        """Generates plots using _create_backtest_plots() helper function.

        Args:
            baseline_start (str, optional): Start date for baseline. Defaults to config["TRADE_START_DATE"].
            baseline_end (str, optional): End date for baseline. Defaults to config["TRADE_END_DATE"].
            baseline_ticker (str, optional): Baseline ticker. Defaults to "^DJI".
            value_col_name (str, optional): Column name in the dataframe for calculating the statistics. Defaults to "account_value".
        """
        for index, portfolio in enumerate(self.portfolio_dfs):
            self._create_backtest_plot(portfolio,
                                       baseline_start,
                                       baseline_end,
                                       baseline_ticker,
                                       value_col_name,
                                       self.agent_names[index])

    @staticmethod
    def _get_daily_return(portfolio_df, value_col_name="account_value", output_col_name="daily_return"):
        """Gets daily return

        Args:
            portfolio_df (pd.DataFrame): portfolio data frame
            value_col_name (str, optional): Column name in the dataframe for calculating the statistics. Defaults to "account_value".
            output_col_name (str, optional): Output column name. Defaults to "Agent".

        Returns:
            pd.Series: daily return
        """
        df = portfolio_df.copy()
        df[output_col_name] = df[value_col_name].pct_change(1)
        df["date"] = pd.to_datetime(df["date"])
        df.set_index("date", inplace=True, drop=True)
        df.index = df.index.tz_localize("UTC")
        return pd.Series(df[output_col_name], index=df.index, dtype='float64')

    @staticmethod
    def _get_baseline(ticker, start, end):
        """Downloads data for the baseline ticker via Yahoo Finance API

        Args:
            ticker (str): baseline ticker
            start (str): start date
            end (str): end date

        Returns:
            pd.DataFrame: downloaded data
        """
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
