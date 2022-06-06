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

    def __init__(self):
        self.df = pd.DataFrame()
        self.df_processed = pd.DataFrame()
        self.custom_df = pd.DataFrame()

    def extend_data(self, original_df, custom_df):
        """Calls the method that adds user defined features to the provided dataframe. 

        Args:
            df (pd.DataFrame): data to be processed

        Returns:
            pd.DataFrame: processed data
        """
        self.df = original_df
        self.df_processed = original_df.copy()
        self.custom_df = custom_df
        self.df_processed = self._add_user_defined_feature()
        return self.df_processed

    # TODO: add user given column
    def _add_user_defined_feature(self):
        """Adds user defined feature column to the current data.
        Args:
            df_column (pd.Series): column to be added.

        Returns:
            pd.DataFrame : Dataframe that has user defined features.
        """
        merged_df = pd.merge(self.df[["date"]], self.custom_df, how="left", on="date")
        self.df_processed = merged_df
        return merged_df

