from sklearn.model_selection import TimeSeriesSplit
import numpy as np
from sklearn.utils import indexable
from sklearn.utils.validation import _num_samples


class BlockingTimeSeriesSplitter(TimeSeriesSplit):
    """Provides methods for implementing Blocking Time Series Splitter.

    Attributes
        n_splits: int
            number of splits
        test_size: int
            size of the test set
        gap: int
            Number of samples to exclude from the end of each train set before the test set.
    Methods
    -------
        _iter_test_indices()
            generates integer indices corresponding to test sets.
        get_n_split()
            returns the number of splitting iterations in the cross-validator
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

    def __init__(self, n_splits=5, *, test_size=None, gap=0):
        super().__init__(n_splits, test_size=test_size, gap=gap)

    def get_n_splits(self, X=None, y=None, groups=None):
        """Returns the number of splitting iterations in the cross-validator

        Args:
            X (object, optional): Always ignored, exists for compatibility. Defaults to None.
            y (object, optional): Always ignored, exists for compatibility. Defaults to None.
            groups (object, optional): Always ignored, exists for compatibility. Defaults to None.

        Returns:
            int : number of splitting iterations in the cross-validator
        """
        return self.n_splits

    def split(self, X, y=None, groups=None):
        """Generate indices to split data into training and test set.

        Args:
            X (array-like of shape (n_samples, n_features)): Training data, where n_samples is the number of samples and n_features is the number of features.
            y (array-like of shape (n_samples)): Always ignored, exists for compatibility. Defaults to None.
            groups (array-like of shape (n_samples) optional): Always ignored, exists for compatibility. Defaults to None.

        Raises:
            ValueError: Raised when the number of folds are greater than number of samples.
            ValueError: Raised when the number of splits is too many for number of samples.
            ValueError: Raised when the number of slipts is incompatible.

        Yields:
            ndarray: The training set indices for that split.
            ndarray: The testing set indices for that split.
        """
        X, y, groups = indexable(X, y, groups)
        n_samples = _num_samples(X)
        n_splits = self.n_splits
        n_folds = n_splits + 1
        gap = self.gap
        test_size = (
            self.test_size if self.test_size is not None else n_samples // n_folds
        )
        # print("test_size:", str(test_size))

        # Make sure we have enough samples for the given split parameters
        if n_folds > n_samples:
            raise ValueError(
                f"Cannot have number of folds={n_folds} greater"
                f" than the number of samples={n_samples}."
            )
        if n_samples - n_splits * (test_size + gap) < n_splits:
            raise ValueError(
                f"Too many splits={n_splits} for number of samples"
                f"={n_samples} with test_size={test_size} and gap={gap}."
            )
        train_size = (n_samples - n_splits * (test_size + gap)) // n_splits
        if (n_samples - n_splits * (test_size + gap)) % n_splits != 0:
            raise ValueError(
                f"Incompatible number of splits={n_splits}, test_size={test_size} and gap={gap}"
            )

        indices = np.arange(n_samples)

        train_starts = range(0, n_samples, train_size + gap + test_size)
        # print("train_size:", str((n_samples - n_splits * (test_size + gap)) // n_splits))
        for train_start in train_starts:
            test_start = train_start + train_size + gap
            yield (
                indices[train_start: train_start + train_size],
                indices[test_start: test_start + test_size]
            )
