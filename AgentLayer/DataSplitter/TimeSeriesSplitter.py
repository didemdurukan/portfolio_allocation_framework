from sklearn.model_selection import TimeSeriesSplit


class TimeSeriesSplitter(TimeSeriesSplit):

    def _iter_test_indices(self, X=None, y=None, groups=None):
        super()._iter_test_indices(X, y, groups)

    def __init__(self, n_splits=5, max_train_size=None, test_size=None, gap=0):
        super().__init__(n_splits, max_train_size=max_train_size, test_size=test_size, gap=gap)

    def split(self, X, y=None, groups=None):
        """Generate indices to split data into training and test set.
          Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Training data, where `n_samples` is the number of samples
            and `n_features` is the number of features.
        y : array-like of shape (n_samples,)
            Always ignored, exists for compatibility.
        groups : array-like of shape (n_samples,)
            Always ignored, exists for compatibility.
        """
        return super().split(X, y, groups)

    @staticmethod
    def get_split_data(df, start, end, target_date_col="date"):
        """
        split the dataset into training or testing using date
        :param data: (df) pandas dataframe, start, end
        :return: (df) pandas dataframe
        @param target_date_col:
        @param end:
        @param start:
        @param df:
        """
        data = df[(df[target_date_col] >= start) & (df[target_date_col] < end)]
        data = data.sort_values([target_date_col, "tic"], ignore_index=True)
        data.index = data[target_date_col].factorize()[0]
        return data
