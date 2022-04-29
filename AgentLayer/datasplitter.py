from sklearn.model_selection import TimeSeriesSplit
import numpy as np
from sklearn.utils import indexable
from sklearn.utils.validation import _num_samples


class ExpandingWindowSplitter(TimeSeriesSplit):

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


class BlockingTimeSeriesSplitter(TimeSeriesSplit):
    def _iter_test_indices(self, X=None, y=None, groups=None):
        super()._iter_test_indices(X, y, groups)

    def __init__(self, n_splits=5, *, test_size=None, gap=0):
        super().__init__(n_splits, test_size=test_size, gap=gap)

    def get_n_splits(self, X=None, y=None, groups=None):
        return self.n_splits

    def split(self, X, y=None, groups=None):
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
