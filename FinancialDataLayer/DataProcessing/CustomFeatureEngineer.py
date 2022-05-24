import pandas as pd

from FinancialDataLayer.DataProcessing.FeatureEngineer import FeatureEngineer


# inspired by:https://stable-baselines.readthedocs.io/en/master/guide/custom_policy.html
# or directly add column?
class CustomFeatureEngineer(FeatureEngineer):

    def __init__(self, df_column):
        self.df_column = df_column
        self.df = pd.DataFrame()
        self.df_processed = pd.DataFrame()

    def extend_data(self, df,):
        self.df = df
        self.df_processed = df.copy()
        self.df_processed = self._add_user_defined_feature(self.df_column)
        return self.df_processed

    def _add_user_defined_feature(self, df_column):  # TODO: add user given column
        """
         add user defined features
        :param data: (df) pandas dataframe
        :return: (df) pandas dataframe
        """
        # df = self.df_processed.copy()
        # for i in range(1, self.lag + 1):
        #     df[f"return_lag_{i}"] = df.close.pct_change(i)
        # df = df.dropna()
        return df
