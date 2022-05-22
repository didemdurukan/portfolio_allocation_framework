import itertools

import numpy as np
import pandas as pd
from stockstats import StockDataFrame as Sdf

import config
from FinancialDataLayer.DataCollection.DataDownloader import DataDownloader
from FinancialDataLayer.DataProcessing.FeatureEngineer import FeatureEngineer


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

    """

    def __init__(self,
                 tech_indicator_list=None,
                 use_default=True,
                 use_covar=False,
                 lookback=252,
                 use_vix=False,
                 use_turbulence=False
                 ):  # TODO: add instance variables here and make other functions member functions (not staticmethods), get rid of __ and instead use _

        if tech_indicator_list is not None and use_default is True:
            raise ValueError("Use default cannot be True if technical indicator list is supplied.")
        if use_default is True:
            self.tech_indicator_list = config.IMPLEMENTED_TECH_INDICATORS_LIST
        else:
            self.tech_indicator_list = tech_indicator_list
        self.use_default = use_default
        self.use_covar = use_covar
        self.lookback = lookback
        self.use_vix = use_vix
        self.use_turbulence = use_turbulence
        self.df = pd.DataFrame()
        self.df_processed = pd.DataFrame()

    def extend_data(self, df):
        """
        main method to do the feature engineering
        @param df:
        @return:
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

        # fill the missing values at the beginning and the end
        self.df_processed = self.df_processed.fillna(method="ffill").fillna(method="bfill")
        # Index - Date Match by DOGAN
        self.df_processed.index = self.df_processed["date"].factorize()[0]

        return self.df_processed

    def _clean_data(self):  # removes delisted
        """
        clean the raw data
        deal with missing values
        reasons: stocks could be delisted, not incorporated at the time step
        :param data: (df) pandas dataframe
        :return: (df) pandas dataframe
        """
        df = self.df_processed.copy()
        df = df.sort_values(["date", "tic"], ignore_index=True)
        df.index = df.date.factorize()[0]  # Turns the index into integers corresponding to `unique` dates
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
        """
        calculate technical indicators
        use stockstats package to add technical inidactors
        :param data: (df) pandas dataframe
        :return: (df) pandas dataframe
        """
        df = self.df_processed.copy()
        df = df.sort_values(by=["tic", "date"])
        stock = Sdf.retype(df.copy())
        unique_ticker = stock.tic.unique()

        for indicator in self.tech_indicator_list:
            indicator_df = pd.DataFrame()
            for i in range(len(unique_ticker)):
                try:
                    temp_indicator = stock[stock.tic == unique_ticker[i]][indicator]
                    temp_indicator = pd.DataFrame(temp_indicator)
                    temp_indicator["tic"] = unique_ticker[i]
                    temp_indicator["date"] = df[df.tic == unique_ticker[i]][
                        "date"
                    ].to_list()
                    indicator_df = pd.concat([indicator_df, temp_indicator], ignore_index=True)
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
        """
        add vix from yahoo finance
        :param data: (df) pandas dataframe
        :return: (df) pandas dataframe
        """
        df = self.df_processed.copy()
        df_vix = DataDownloader(start_date=df.date.min(), end_date=df.date.max(), ticker_list=["^VIX"]).collect()
        vix = df_vix[["date", "close"]]
        vix.columns = ["date", "vix"]

        df = df.merge(vix, on="date")
        df = df.sort_values(["date", "tic"]).reset_index(drop=True)
        return df

    def _add_turbulence(self):
        """
        add turbulence index from a precalculated dataframe
        :param data: (df) pandas dataframe
        :return: (df) pandas dataframe
        """
        df = self.df_processed.copy()
        turbulence_index = self._calculate_turbulence()
        df = df.merge(turbulence_index, on="date")
        df = df.sort_values(["date", "tic"]).reset_index(drop=True)
        return df

    def _calculate_turbulence(self):
        """calculate turbulence index based on dow 30
        @return:
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
            current_price = df_price_pivot[df_price_pivot.index == unique_date[i]]
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
        """

        @return:
        """
        df = self.df_processed.copy()
        # TODO: Check if some of these preprocessing steps are necessary
        df['date'] = df['date'].astype(str)  # convert to string temporarily for concatenation purposes
        ticker_list = df["tic"].unique().tolist()  # get ticker types
        date_list = list(pd.date_range(df['date'].min(), df['date'].max()).astype(str))  # get dates
        date_ticker_list = list(itertools.product(date_list, ticker_list))  # combine them
        df_processed_full = pd.DataFrame(date_ticker_list, columns=["date", "tic"]).merge(df,
                                                                                          on=["date", "tic"],
                                                                                          how="left")  # apply left join with that combination
        df_processed_full['date'] = pd.to_datetime(df_processed_full['date'],
                                                   format='%Y-%m-%d')  # back to datetime format
        df_processed_full = df_processed_full[
            df_processed_full['date'].isin(df['date'])]  # keep only actual data by matching the dates
        df_processed_full = df_processed_full.sort_values(['date', 'tic'])  # sort by date-ticker combination
        df_processed_full = df_processed_full.fillna(
            0)  # fill the missing data with 0 # TODO: Check if this is a good idea

        # include covariance of stocks as feature depending on 1 year data
        df_processed_full = df_processed_full.sort_values(['date', 'tic'], ignore_index=True)
        df_processed_full.index = df_processed_full.date.factorize()[0]
        cov_list = []
        return_list = []
        # look back is one year
        lookback = self.lookback
        for i in range(lookback, len(df_processed_full.index.unique())):
            data_lookback = df_processed_full.loc[i - lookback:i, :]
            price_lookback = data_lookback.pivot_table(index='date', columns='tic', values='close')
            return_lookback = price_lookback.pct_change().dropna()
            return_list.append(return_lookback)
            covs = return_lookback.cov().values
            cov_list.append(covs)

        df_cov = pd.DataFrame(
            {'date': df_processed_full.date.unique()[lookback:], 'cov_list': cov_list, 'return_list': return_list})

        df_processed_full = df_processed_full.merge(df_cov, on='date')
        df_processed_full = df_processed_full.sort_values(['date', 'tic']).reset_index(drop=True)
        return df_processed_full

    def prepare_ml_data(self, train_data):
        """

        @param train_data:
        @return:
        """
        train_date = sorted(set(train_data.date.values))
        X = []
        for i in range(0, len(train_date) - 1):
            d = train_date[i]
            d_next = train_date[i + 1]
            # TODO: if check for existence of return_list, if doesnt exist add by default then continue
            y = train_data.loc[train_data['date'] == d_next].return_list.iloc[0].loc[d_next].reset_index()
            y.columns = ['tic', 'return']
            x = train_data.loc[train_data['date'] == d][self.tech_indicator_list]
            train_piece = pd.merge(x, y, on='tic')
            train_piece['date'] = [d] * len(train_piece)
            X += [train_piece]
        train_data_ml = pd.concat(X)
        X = train_data_ml[self.tech_indicator_list].values
        Y = train_data_ml[['return']].values

        return X, Y

