import pandas as pd

from FinancialDataLayer.DataProcessing.FeatureEngineer import FeatureEngineer

# inspired by:https://stable-baselines.readthedocs.io/en/master/guide/custom_policy.html


class CustomFeatureEngineer(FeatureEngineer):
    """Provides methods for applying custom feature engineering to
    the current dataframe. 

    Attributes
    ----------
        df_column: pd.Series
            column to be concatanated to the dataframe as a new a feature.
        df: pd.DataFrame
            dataframe to be processed
        df_processed: pd.DataFrame
            processed dataframe

    Methods
    -------
        extend_data()
            main method that calls the function that adds custom features
        _add_user_defined_feature()
            adds user defined features to the dataframe.

    """

    def __init__(self, df_column):
        self.df_column = df_column
        self.df = pd.DataFrame()
        self.df_processed = pd.DataFrame()

    def extend_data(self, df,):
        """Calls the method that adds user defined features to the provided dataframe. 

        Args:
            df (pd.DataFrame): data to be processed

        Returns:
            pd.DataFrame: processed data
        """
        self.df = df
        self.df_processed = df.copy()
        self.df_processed = self._add_user_defined_feature(self.df_column)
        return self.df_processed

    # TODO: add user given column
    def _add_user_defined_feature(self, df_column):
        """Adds user defined feature column to the current data.
        Args:
            df_column (pd.Series): column to be added.

        Returns:
            pd.DataFrame : Dataframe that has user defined features.
        """
        # df = self.df_processed.copy()
        # for i in range(1, self.lag + 1):
        #     df[f"return_lag_{i}"] = df.close.pct_change(i)
        # df = df.dropna()
        return df
