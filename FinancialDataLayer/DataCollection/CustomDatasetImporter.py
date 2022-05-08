from FinancialDataLayer.DataCollection.DataCollector import DatasetCollector, load_from_file, load_from_df


class CustomDatasetImporter(DatasetCollector):

    def collect(self):
        if self.df is None:
            df = load_from_file(self.path)
        else:
            df = load_from_df(self.df)
        self.df = df
        return self.df

    def __init__(self, path="", df=None):
        self.df = df
        self.path = path