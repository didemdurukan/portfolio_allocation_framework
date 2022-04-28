import config
import pandas as pd
from datacollector import DataDownloader
import numpy as np
from stockstats import StockDataFrame as Sdf


# TODO: class DataCleaner:


class FeatureEngineer:
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
        main method to do the feature engineering

    """

    def __init__(self):
        pass

    @staticmethod
    def add_features(df,
                     use_default: bool = True,
                     tech_indicator_list: list = config.IMPLEMENTED_TECH_INDICATORS_LIST,
                     use_vix: bool = False,
                     use_turbulence: bool = False,
                     user_defined_feature: bool = False):

        extended_df = FeatureEngineer.__extend_data(df,
                                                    use_default,
                                                    tech_indicator_list,
                                                    use_vix,
                                                    use_turbulence,
                                                    user_defined_feature)
        return extended_df

    @staticmethod
    def __extend_data(df,
                      use_default,
                      tech_indicator_list,
                      use_vix,
                      use_turbulence,
                      user_defined_feature):
        """
        main method to do the feature engineering
        @param df:
        @return:
        """
        # clean data
        df = FeatureEngineer.__clean_data(df)

        # add technical indicators using stockstats
        if use_default:
            df = FeatureEngineer.__add_technical_indicator(df, config.IMPLEMENTED_TECH_INDICATORS_LIST)
            print("Successfully added technical indicators")
        elif not use_default:
            df = FeatureEngineer.__add_technical_indicator(df, tech_indicator_list)
            print("Successfully added technical indicators")

        # add vix for multiple stock
        if use_vix:
            df = FeatureEngineer.__add_vix(df)
            print("Successfully added vix")

        # add turbulence index for multiple stock
        if use_turbulence:
            df = FeatureEngineer.__add_turbulence(df)
            print("Successfully added turbulence index")

        # add user defined feature
        if user_defined_feature:
            df = FeatureEngineer.__add_user_defined_feature(df)
            print("Successfully added user defined features")

        # fill the missing values at the beginning and the end
        df = df.fillna(method="ffill").fillna(method="bfill")
        return df

    @staticmethod
    def __clean_data(data):
        """
        clean the raw data
        deal with missing values
        reasons: stocks could be delisted, not incorporated at the time step
        :param data: (df) pandas dataframe
        :return: (df) pandas dataframe
        """
        df = data.copy()
        df = df.sort_values(["date", "tic"], ignore_index=True)
        df.index = df.date.factorize()[0]
        merged_closes = df.pivot_table(index="date", columns="tic", values="close")
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

    @staticmethod
    def __add_technical_indicator(data, tech_indicator_list):
        """
        calculate technical indicators
        use stockstats package to add technical inidactors
        :param data: (df) pandas dataframe
        :return: (df) pandas dataframe
        """
        df = data.copy()
        df = df.sort_values(by=["tic", "date"])
        stock = Sdf.retype(df.copy())
        unique_ticker = stock.tic.unique()

        for indicator in tech_indicator_list:
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

    @staticmethod
    def __add_user_defined_feature(data):
        """
         add user defined features
        :param data: (df) pandas dataframe
        :return: (df) pandas dataframe
        """
        df = data.copy()
        df["daily_return"] = df.close.pct_change(1)
        # df['return_lag_1']=df.close.pct_change(2)
        # df['return_lag_2']=df.close.pct_change(3)
        # df['return_lag_3']=df.close.pct_change(4)
        # df['return_lag_4']=df.close.pct_change(5)
        return df

    @staticmethod
    def __add_vix(data):
        """
        add vix from yahoo finance
        :param data: (df) pandas dataframe
        :return: (df) pandas dataframe
        """
        df = data.copy()
        df_vix = DataDownloader.download_data(start_date=df.date.min(), end_date=df.date.max(), ticker_list=["^VIX"])
        vix = df_vix[["date", "close"]]
        vix.columns = ["date", "vix"]

        df = df.merge(vix, on="date")
        df = df.sort_values(["date", "tic"]).reset_index(drop=True)
        return df

    @staticmethod
    def __add_turbulence(data):
        """
        add turbulence index from a precalculated dataframe
        :param data: (df) pandas dataframe
        :return: (df) pandas dataframe
        """
        df = data.copy()
        turbulence_index = FeatureEngineer.__calculate_turbulence(df)
        df = df.merge(turbulence_index, on="date")
        df = df.sort_values(["date", "tic"]).reset_index(drop=True)
        return df

    @staticmethod
    def __calculate_turbulence(data):
        """calculate turbulence index based on dow 30"""
        # can add other market assets
        df = data.copy()
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
