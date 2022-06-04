from sklearn.model_selection import TimeSeriesSplit


class TimeSeriesSplitter(TimeSeriesSplit):
    """Provides methods for implementing Time Series Splitter.

    Attributes
        n_splits: int
            number of splits
        max_train_size: int
            maximum train size
        test_size: int
            size of the test set
        gap: int
            Number of samples to exclude from the end of each train set before the test set.
    Methods
    -------
        _iter_test_indices()
            generates integer indices corresponding to test sets.
        get_split_data()
            split the dataset into training or testing using date
        split()
            generate indices to split data into training and test set
    """

    def _iter_test_indices(self, X=None, y=None, groups=None):
        """Generates integer indices corresponding to test sets.

        Args:
            X (object, optional): Always ignored, exists for compatibility. Defaults to None.
            y (object, optional): Always ignored, exists for compatibility. Defaults to None.
            groups (object, optional): Always ignored, exists for compatibility. Defaults to None.
        """
        super()._iter_test_indices(X, y, groups)

    def __init__(self, n_splits=5, max_train_size=None, test_size=None, gap=0):
        super().__init__(n_splits, max_train_size=max_train_size, test_size=test_size, gap=gap)

    def split(self, X, y=None, groups=None):
        """Generate indices to split data into training and test set.
          Parameters

        Args:
            X (array-like of shape (n_samples, n_features)): Training data, where n_samples is the number of samples and n_features is the number of features.
            y (array-like of shape (n_samples)): Always ignored, exists for compatibility. Defaults to None.
            groups (array-like of shape (n_samples) optional): Always ignored, exists for compatibility. Defaults to None.
        """
        return super().split(X, y, groups)

    @staticmethod
    def get_split_data(df, start, end, target_date_col="date"):
        """_summary_

        Args:
            df (pd.DataFrame): Dataframe to be splitted
            start (str): start date
            end (str): end date
            target_date_col (str, optional): target column. Defaults to "date".

        Returns:
            pd.DataFrame: splitted data
        """
        data = df[(df[target_date_col] >= start) & (df[target_date_col] < end)]
        data = data.sort_values([target_date_col, "tic"], ignore_index=True)
        data.index = data[target_date_col].factorize()[0]
        return data
