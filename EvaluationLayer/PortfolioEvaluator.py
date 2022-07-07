from matplotlib import pyplot as plt
from pyfolio import timeseries
import pyfolio
import pandas as pd
from FinancialDataLayer.DataCollection.DataDownloader import DataDownloader
from EvaluationLayer.Evaluator import Evaluator
import warnings
from utils import read_config_file

warnings.simplefilter(action='ignore', category=FutureWarning)
config = read_config_file()


class PortfolioEvaluator(Evaluator):
    """Provides methods for portfolio evaluation.

        Attributes
        ----------
            -portfolio_dfs : pd.DataFrame
                portfolios to be evaluated.
            -agent_names : list
                agent names
            -stats_list : list
                list for statistics.
            -benchmark_start : str
                start date for downloading data for the benchmark ticker
            -benchmark_end : str
                end date for downloading data for the benchmark ticker
            -benchmark_ticker : str
                benchmark ticker
            -benchmark-df : pd.DataFrame

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
            -_get_benchmark()
                downloads the data of the benchmark from Yahoo API

    """

    def __init__(self,
                 *portfolio_dfs,
                 agent_names,
                 benchmark_start=config["TRADE_START_DATE"],
                 benchmark_end=config["TRADE_END_DATE"],
                 benchmark_ticker=config["BENCHMARK_TICKER"]):
        """Initiliazer for PortfolioEvaluator object.

        Args:
            portfolio_dfs (pd.DataFrame) : Portfolio Dataframes.
            agent_names (list): list of agent names
            benchmark_start (str, optional): start date for downloading data for the benchmark ticker. Defaults to config["TRADE_START_DATE"].
            benchmark_end (str, optional): end date for downloading data for the benchmark ticker. Defaults to config["TRADE_END_DATE"].
            benchmark_ticker (str, optional): benchmark ticker. Defaults to config["benchmark_TICKER"].
        """
        self.portfolio_dfs = portfolio_dfs
        if agent_names is None:
            self.agent_names = ["Agent" + str(i)
                                for i in range(len(portfolio_dfs))]
        else:
            self.agent_names = agent_names
        self.stats_list = None
        self.benchmark_start = benchmark_start
        self.benchmark_end = benchmark_end
        self.benchmark_ticker = benchmark_ticker
        self.benchmark_df = None

    def backtest_stats(self,
                       value_col_name="account_value"):
        """Gets backtest statistics using _get_portfolio_stats() and _get_benchmark_stats helper functions.

        Args:
            value_col_name (str, optional): Column name in the dataframe for calculating the portfolio statistics. Defaults to "account_value".
        Returns:
            pd.DataFrame: backtest statistics
        """
        perf_stats_list = []
        index_list = self.agent_names.copy()
        for portfolio in self.portfolio_dfs:
            perf_stats_list.append(
                self._get_portfolio_stats(portfolio, value_col_name))
        if self.benchmark_ticker is not None:
            benchmark_stats = self._get_benchmark_stats()
            perf_stats_list.append(benchmark_stats.copy())
            index_list.append(self.benchmark_ticker)
        self.stats_list = perf_stats_list.copy()
        stats_df = pd.DataFrame(perf_stats_list, index=index_list)
        return stats_df

    def _get_portfolio_stats(self, portfolio_df, value_col_name="account_value"):
        """Calculates portfolio statistics.

        Args:
            portfolio_df (pd.DataFrame): portfolio data frame
            value_col_name (str, optional): Column name in the dataframe for calculating the portfolio statistics. Defaults to "account_value".

        Returns:
            pd.Series : Performance statistics.
        """
        df = portfolio_df.copy()
        dr_test = self._get_daily_return(
            df, value_col_name=value_col_name)
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
                              value_col_name="account_value",
                              agent_label="Agent"):
        """Creates backtest plots

        Args:
            portfolio_df (pd.DataFrame): portfolio data frame
            value_col_name (str, optional): Column name in the dataframe for calculating the statistics. Defaults to "account_value".
            agent_label (str, optional): Agent label in the plot. Defaults to "Agent".
        """
        df = portfolio_df.copy()
        df["date"] = pd.to_datetime(df["date"])
        test_returns = self._get_daily_return(
            df, value_col_name=value_col_name)
        test_returns.name = agent_label  # agent label determines the label for the agent
        print("Stats for agent:", agent_label)
        print("*********************************************************************************")
        benchmark_df = self.benchmark_df if self.benchmark_df is not None else self._get_benchmark()
        self.benchmark_df = benchmark_df.copy()

        benchmark_df["date"] = pd.to_datetime(
            benchmark_df["date"], format="%Y-%m-%d")
        benchmark_df = pd.merge(
            df[["date"]], benchmark_df, how="left", on="date")
        benchmark_df = benchmark_df.fillna(method="ffill").fillna(method="bfill")
        benchmark_returns = self._get_daily_return(
            benchmark_df, value_col_name="close")
        benchmark_returns.name = self.benchmark_ticker
        # with pyfolio.plotting.plotting_context(font_scale=1.1):
        #     pyfolio.create_full_tear_sheet(
        #         returns=test_returns, benchmark_rets=benchmark_returns
        #     )
        #
        # define plotting region (1 row, 2 columns)
        plt.rcParams.update({'axes.titlesize': 'large',
                             'axes.labelsize': 'large',
                             'ytick.labelsize': 'large',
                             'xtick.labelsize': 'large',
                             'figure.figsize': [15, 5]})
        with pyfolio.plotting.plotting_context(font_scale=5, rc=plt.rcParams):
            # pyfolio.create_full_tear_sheet(
            #     returns=test_returns, benchmark_rets=benchmark_returns, set_context=True
            # )
            pyfolio.create_returns_tear_sheet(returns=test_returns, benchmark_rets=benchmark_returns, set_context=False)

    def _get_benchmark_stats(self):
        """Calculate benchmark statistics

        Args:
            output_col_name (str, optional): Output column in the dataframe. Defaults to "daily_return".

        Returns:
            pd.Series : Performance statistics.
        """

        benchmark_df = self.benchmark_df if self.benchmark_df is not None else self._get_benchmark()
        self.benchmark_df = benchmark_df.copy()
        benchmark_df["date"] = pd.to_datetime(
            benchmark_df["date"], format="%Y-%m-%d")
        benchmark_df = benchmark_df.fillna(method="ffill").fillna(method="bfill")
        benchmark_returns = self._get_daily_return(
            benchmark_df, value_col_name="close")
        benchmark_perf_stats = timeseries.perf_stats(
            returns=benchmark_returns,
            factor_returns=None,
            positions=None,
            transactions=None,
            turnover_denom="AGB"
        )
        return benchmark_perf_stats

    def backtest_plot(self, value_col_name="account_value"):
        """Generates plots using _create_backtest_plots() helper function.

        Args:
            value_col_name (str, optional): Column name in the dataframe for calculating the statistics. Defaults to "account_value".
        """
        for index, portfolio in enumerate(self.portfolio_dfs):
            df = portfolio.copy()
            self._create_backtest_plot(df,
                                       value_col_name,
                                       self.agent_names[index])

    @staticmethod
    def _get_daily_return(portfolio_df, value_col_name="account_value"):
        """Gets daily return

        Args:
            portfolio_df (pd.DataFrame): portfolio data frame
            value_col_name (str, optional): Column name in the dataframe for calculating the statistics. Defaults to "account_value".

        Returns:
            pd.Series: daily return
        """
        df = portfolio_df.copy()
        df["daily_return"] = df[value_col_name].pct_change(1)
        df["date"] = pd.to_datetime(df["date"])
        df.set_index("date", inplace=True, drop=True)
        df.index = df.index.tz_localize("UTC")
        returns = pd.Series(df["daily_return"], index=df.index, dtype='float64').dropna()
        return returns

    def _get_benchmark(self):
        """Gets benchmark ticker data via Yahoo Finance API

        Returns:
            pd.DataFrame: Data for the benchmark ticker.
        """
        return DataDownloader(start_date=self.benchmark_start, end_date=self.benchmark_end,
                              ticker_list=[self.benchmark_ticker]).download_from_yahoo()
