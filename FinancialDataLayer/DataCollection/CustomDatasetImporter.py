from FinancialDataLayer.DataCollection.DatasetCollector import DatasetCollector


class CustomDatasetImporter(DatasetCollector):

    def collect(self):
        if self.df is None:
            df = self.load_from_file(self.path)
        else:
            df = self.load_from_df(self.df)
        self.df = df
        return self.df

    def __init__(self, path="", df=None):
        self.df = df
        self.path = path
