import itertools

import numpy as np
import pandas as pd
from stockstats import StockDataFrame as Sdf
from FinancialDataLayer.DataCollection.DataDownloader import DataDownloader
from FinancialDataLayer.DataProcessing.FeatureEngineer import FeatureEngineer
import yaml

config = yaml.safe_load(open("user_params.yaml"))
# config = yaml.safe_load(open("user_params.yaml")) #bende boyleyken calisiyor


class DefaultFeatureEngineer(FeatureEngineer):
    """Provides methods for feature engineer to apply to security price data

    Attributes
    ----------
        tech_indicator_list: list
            a list of technical indicator names
        use_turbulence: boolean
            use turbulence index or not
        user_defined_feature: boolean
            user defined features or not

    Methods
    -------
        extend_data()
            main method that does feature engineering 
        _clean_data()
            cleans the raw data and deals with missing values
        _add_technical_indicator()
            calculates technical indicators and uses stockstats package to add technical inidactors.
        _add_vix()
            adds vix from Yahoo Finance.
        _add_turbulence()
            adds turbulence index from a precalculated dataframe
        _calculate_turbulence()
            calculate turbulence index based on dow 30
        __add_covariances()
            adds covariences to the dataframe
        _add_returns()
            adds returns to the dataframe
        prepare_ml_data()
            splits into train_X and train_Y


    """

    def __init__(self,
                 tech_indicator_list=None,
                 use_default=True,
                 use_covar=True,
                 use_return=True,
                 lookback=252,
                 use_vix=False,
                 use_turbulence=False
                 ):

        if tech_indicator_list is not None and use_default is True:
            raise ValueError(
                "Use default cannot be True if technical indicator list is supplied.")
        if use_default is True:
            self.tech_indicator_list = config["feature_eng_params"]["tech_indicator_list"]
        else:
            self.tech_indicator_list = tech_indicator_list
        self.use_default = use_default
        self.use_covar = use_covar
        self.use_return = use_return
        self.lookback = lookback
        self.use_vix = use_vix
        self.use_turbulence = use_turbulence
        self.df = pd.DataFrame()
        self.df_processed = pd.DataFrame()

    def extend_data(self, df):
        """Main method to do the feature engineering

        Args:
            df (pd.DataFrame) : dataframe to be processed. 

        Returns:
            pd.DataFrame: processed dataframe. 
        """
        self.df = df
        self.df_processed = df.copy()

        # clean data (deals with missing values and delisted tickers)
        self.df_processed = self._clean_data()

        # add technical indicators using stockstats
        if self.use_default:
            self.df_processed = self._add_technical_indicator()
            print("Successfully added technical indicators")

        elif not self.use_default:
            self.df_processed = self._add_technical_indicator()
            print("Successfully added technical indicators")

        # add vix for multiple stock (volatility index)
        if self.use_vix:
            self.df_processed = self._add_vix()
            print("Successfully added vix")

        # add turbulence index for multiple stock
        if self.use_turbulence:
            self.df_processed = self._add_turbulence()
            print("Successfully added turbulence index")

        # add covariances
        if self.use_covar:
            self.df_processed = self._add_covariances()
            print("Successfully added covariances")

        # add covariances
        if self.use_return:
            self.df_processed = self._add_returns()
            print("Successfully added returns")

        # fill the missing values at the beginning and the end
        self.df_processed = self.df_processed.fillna(
            method="ffill").fillna(method="bfill")
        # Index - Date Match by DOGAN
        self.df_processed.index = self.df_processed["date"].factorize()[0]

        return self.df_processed

    def _clean_data(self):  # removes delisted
        """Cleans the raw data and deals with missing values

        Returns:
            pd.DataFrame : Cleaned dataframe. 
        """
        df = self.df_processed.copy()
        df = df.sort_values(["date", "tic"], ignore_index=True)
        # Turns the index into integers corresponding to `unique` dates
        df.index = df.date.factorize()[0]
        merged_closes = df.pivot_table(index="date", columns="tic",
                                       values="close")  # an excel style pivot table that produces ticker columns and values as close
        merged_closes = merged_closes.dropna(axis=1)
        tics = merged_closes.columns
        df = df[df.tic.isin(tics)]
        # df = data.copy()
        # list_ticker = df["tic"].unique().tolist()
        # only apply to daily level data, need to fix for minute level
        # list_date = list(pd.date_range(df['date'].min(),df['date'].max()).astype(str))
        # combination = list(itertools.product(list_date,list_ticker))

        # df_full = pd.DataFrame(combination,columns=["date","tic"]).merge(df,on=["date","tic"],how="left")
        # df_full = df_full[df_full['date'].isin(df['date'])]
        # df_full = df_full.sort_values(['date','tic'])
        # df_full = df_full.fillna(0)
        return df

    def _add_technical_indicator(self):
        """Calculates technical indicators and uses stockstats package to add technical inidactors

        Returns:
            pd.DataFrame : Dataframe that contains technical indicators.
        """
        df = self.df_processed.copy()
        df = df.sort_values(by=["tic", "date"])
        stock = Sdf.retype(df.copy())
        unique_ticker = stock.tic.unique()

        for indicator in self.tech_indicator_list:
            indicator_df = pd.DataFrame()
            for i in range(len(unique_ticker)):
                try:
                    temp_indicator = stock[stock.tic ==
                                           unique_ticker[i]][indicator]
                    temp_indicator = pd.DataFrame(temp_indicator)
                    temp_indicator["tic"] = unique_ticker[i]
                    temp_indicator["date"] = df[df.tic == unique_ticker[i]][
                        "date"
                    ].to_list()
                    indicator_df = pd.concat(
                        [indicator_df, temp_indicator], ignore_index=True)
                except Exception as e:
                    print(e)
            df = df.merge(
                indicator_df[["tic", "date", indicator]], on=["tic", "date"], how="left"
            )
        df = df.sort_values(by=["date", "tic"])
        return df
        # df = data.set_index(['date','tic']).sort_index()
        # df = df.join(df.groupby(level=0, group_keys=False).apply(lambda x, y: Sdf.retype(x)[y], y=self.tech_indicator_list))
        # return df.reset_index()

    def _add_vix(self):
        """Adds vix from yahoo finance

        Returns:
            pd.DataFrame : Dataframe that contains vix.
        """
        df = self.df_processed.copy()
        df_vix = DataDownloader(start_date=df.date.min(
        ), end_date=df.date.max(), ticker_list=["^VIX"]).collect()
        vix = df_vix[["date", "close"]]
        vix.columns = ["date", "vix"]

        df = df.merge(vix, on="date")
        df = df.sort_values(["date", "tic"]).reset_index(drop=True)
        return df

    def _add_turbulence(self):
        """Adds turbulence index from a precalculated dataframe

        Returns:
            pd.DataFrame : Dataframe that contains turbulence index as a feature.
        """
        df = self.df_processed.copy()
        turbulence_index = self._calculate_turbulence()
        df = df.merge(turbulence_index, on="date")
        df = df.sort_values(["date", "tic"]).reset_index(drop=True)
        return df

    def _calculate_turbulence(self):
        """Calculates turbulence index based on dow 30

        Returns:
            pd.DataFrame : turbulence index.
        """
        # can add other market assets
        df = self.df_processed.copy()
        df_price_pivot = df.pivot(index="date", columns="tic", values="close")
        # use returns to calculate turbulence
        df_price_pivot = df_price_pivot.pct_change()

        unique_date = df.date.unique()
        # start after a year
        start = 252
        turbulence_index = [0] * start
        # turbulence_index = [0]
        count = 0
        for i in range(start, len(unique_date)):
            current_price = df_price_pivot[df_price_pivot.index ==
                                           unique_date[i]]
            # use one year rolling window to calculate covariance
            hist_price = df_price_pivot[
                (df_price_pivot.index < unique_date[i])
                & (df_price_pivot.index >= unique_date[i - 252])
            ]
            # Drop tickers which has number missing values more than the "oldest" ticker
            filtered_hist_price = hist_price.iloc[
                hist_price.isna().sum().min():
            ].dropna(axis=1)

            cov_temp = filtered_hist_price.cov()
            current_temp = current_price[[x for x in filtered_hist_price]] - np.mean(
                filtered_hist_price, axis=0
            )
            # cov_temp = hist_price.cov()
            # current_temp=(current_price - np.mean(hist_price,axis=0))

            temp = current_temp.values.dot(np.linalg.pinv(cov_temp)).dot(
                current_temp.values.T
            )
            if temp > 0:
                count += 1
                if count > 2:
                    turbulence_temp = temp[0][0]
                else:
                    # avoid large outlier because of the calculation just begins
                    turbulence_temp = 0
            else:
                turbulence_temp = 0
            turbulence_index.append(turbulence_temp)

        turbulence_index = pd.DataFrame(
            {"date": df_price_pivot.index, "turbulence": turbulence_index}
        )
        return turbulence_index

    # TODO: add lookback as parameter
    def _add_covariances(self):
        """Adds covariences as a feature

        Returns:
            pd.DataFrame : Dataframe that contains covariences as a feature.
        """
        df = self.df_processed.copy()
        # TODO: Check if some of these preprocessing steps are necessary
        # convert to string temporarily for concatenation purposes
        df['date'] = df['date'].astype(str)
        ticker_list = df["tic"].unique().tolist()  # get ticker types
        date_list = list(pd.date_range(
            df['date'].min(), df['date'].max()).astype(str))  # get dates
        date_ticker_list = list(itertools.product(
            date_list, ticker_list))  # combine them
        df_processed_full = pd.DataFrame(date_ticker_list, columns=["date", "tic"]).merge(df,
                                                                                          on=[
                                                                                              "date", "tic"],
                                                                                          how="left")  # apply left join with that combination
        df_processed_full['date'] = pd.to_datetime(df_processed_full['date'],
                                                   format='%Y-%m-%d')  # back to datetime format
        df_processed_full = df_processed_full[
            df_processed_full['date'].isin(df['date'])]  # keep only actual data by matching the dates
        df_processed_full = df_processed_full.sort_values(
            ['date', 'tic'])  # sort by date-ticker combination
        df_processed_full = df_processed_full.fillna(
            0)  # fill the missing data with 0 # TODO: Check if this is a good idea

        # include covariance of stocks as feature depending on 1 year data
        df_processed_full = df_processed_full.sort_values(
            ['date', 'tic'], ignore_index=True)
        df_processed_full.index = df_processed_full.date.factorize()[0]
        cov_list = []
        return_list = []
        # look back is one year
        lookback = self.lookback
        for i in range(lookback, len(df_processed_full.index.unique())):
            data_lookback = df_processed_full.loc[i - lookback:i, :]
            price_lookback = data_lookback.pivot_table(
                index='date', columns='tic', values='close')
            return_lookback = price_lookback.pct_change().dropna()
            # return_list.append(return_lookback)
            covs = return_lookback.cov().values
            cov_list.append(covs)

        # df_cov = pd.DataFrame(
        #     {'date': df_processed_full.date.unique()[lookback:], 'cov_list': cov_list, 'return_list': return_list})
        df_cov = pd.DataFrame(
            {'date': df_processed_full.date.unique()[lookback:], 'cov_list': cov_list})

        df_processed_full = df_processed_full.merge(df_cov, on='date')
        df_processed_full = df_processed_full.sort_values(
            ['date', 'tic']).reset_index(drop=True)
        return df_processed_full

    def _add_returns(self):
        """Adds return values to the dataframe

        Returns:
            pd.DataFrame : Dataframe that contains returns values.
        """
        df = self.df_processed.copy()
        # TODO: Check if some of these preprocessing steps are necessary
        # convert to string temporarily for concatenation purposes
        df['date'] = df['date'].astype(str)
        ticker_list = df["tic"].unique().tolist()  # get ticker types
        date_list = list(pd.date_range(
            df['date'].min(), df['date'].max()).astype(str))  # get dates
        date_ticker_list = list(itertools.product(
            date_list, ticker_list))  # combine them
        df_processed_full = pd.DataFrame(date_ticker_list, columns=["date", "tic"]).merge(df,
                                                                                          on=[
                                                                                              "date", "tic"],
                                                                                          how="left")  # apply left join with that combination
        df_processed_full['date'] = pd.to_datetime(df_processed_full['date'],
                                                   format='%Y-%m-%d')  # back to datetime format
        df_processed_full = df_processed_full[
            df_processed_full['date'].isin(df['date'])]  # keep only actual data by matching the dates
        df_processed_full = df_processed_full.sort_values(
            ['date', 'tic'])  # sort by date-ticker combination
        df_processed_full = df_processed_full.fillna(
            0)  # fill the missing data with 0 # TODO: Check if this is a good idea

        # include covariance of stocks as feature depending on 1 year data
        df_processed_full = df_processed_full.sort_values(
            ['date', 'tic'], ignore_index=True)
        df_processed_full.index = df_processed_full.date.factorize()[0]
        return_list = []
        # look back is one year
        lookback = self.lookback
        for i in range(lookback, len(df_processed_full.index.unique())):
            data_lookback = df_processed_full.loc[i - lookback:i, :]
            price_lookback = data_lookback.pivot_table(
                index='date', columns='tic', values='close')
            return_lookback = price_lookback.pct_change().dropna()
            return_list.append(return_lookback)
        df_ret = pd.DataFrame(
            {'date': df_processed_full.date.unique()[lookback:], 'return_list': return_list})

        df_processed_full = df_processed_full.merge(df_ret, on='date')
        df_processed_full = df_processed_full.sort_values(
            ['date', 'tic']).reset_index(drop=True)
        return df_processed_full

    def prepare_ml_data(self, train_data):
        """Splits data to X (features) and y (labels)

        Args:
            train_data (pd.DataFrame): data to be splitted.
        Raises:
            Exception: Raised when provided data does not contain return values
        Returns:
            pd.DataFrame : dataframe that has the features
            pd.DataFrame : dataframe that has the return values. 
        """
        train_date = sorted(set(train_data.date.values))
        X = []
        for i in range(0, len(train_date) - 1):
            d = train_date[i]
            d_next = train_date[i + 1]
            if train_data.get("return_list") is None:
                raise Exception(
                    "return_list not found on train data, please add returns by setting use_return=True")
            y = train_data.loc[train_data['date'] ==
                               d_next].return_list.iloc[0].loc[d_next].reset_index()
            y.columns = ['tic', 'return']
            x = train_data.loc[train_data['date'] ==
                               d][["tic"] + self.tech_indicator_list]
            train_piece = pd.merge(x, y, on='tic')
            train_piece['date'] = [d] * len(train_piece)
            X += [train_piece]
        train_data_ml = pd.concat(X)
        X = train_data_ml[self.tech_indicator_list].values
        Y = train_data_ml[['return']].values

        return X, Y
