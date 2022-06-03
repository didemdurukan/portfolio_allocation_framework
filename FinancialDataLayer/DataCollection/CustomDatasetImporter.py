from FinancialDataLayer.DataCollection.DatasetCollector import DatasetCollector


class CustomDatasetImporter(DatasetCollector):

    def collect(self):
        """Calls the data collection methods according to the
        given parameter. If dataframe is provided then the function 
        that loads data from the dataframe is called, and if the path is
        provided the function that loads the data frame from a file is called.

        Returns:
            pd.DataFrame : loaded data
        """
        if self.df is None:
            df = self.load_from_file(self.path)
        else:
            df = self.load_from_df(self.df)
        self.df = df
        return self.df

    def __init__(self, path="", df=None):
        self.df = df
        self.path = path
