import sklearn as sklearn

'''
Provides metrics to assess performance on regression task.
'''

__ALL__ = [
    "max_error",
    "mean_absolute_error",
    "mean_squared_error",
    "mean_squared_log_error",
    "median_absolute_error",
    "r2_score",
    "explained_variance_score",
    "mean_tweedie_deviance",
    "mean_poisson_deviance",
    "mean_gamma_deviance",
]


def max_error(y_true, y_pred):
    """Implements max_error metric

    Args:
        y_true (pd.DataFrame): Ground truth (correct) target values.
        y_pred (pd.DataFrame): Estimated target values.

    Returns:
        float: max error
    """
    return sklearn.metrics.max_error(y_true, y_pred)


def mean_absolute_error(y_true, y_pred, *, sample_weight=None, multioutput="uniform_average"):
    """Implements mean_absolute_error

    Args:
        y_true (pd.DataFrame): Ground truth (correct) target values.
        y_pred (pd.DataFrame): Estimated target values.
        sample_weight (array-like of shape, optional): Sample weights. Defaults to None.
        multioutput (str, optional): Defines aggregating of multiple output values. Defaults to "uniform_average".

    Returns:
        float: mean absolute error
    """
    return sklearn.metrics.mean_absolute_error(y_true, y_pred, sample_weight=sample_weight, multioutput=multioutput)


def mean_squared_error(y_true, y_pred, *, sample_weight=None, multioutput="uniform_average", squared=True):
    """Implements mean_squared_error

    Args:
        y_true (pd.DataFrame): Ground truth (correct) target values.
        y_pred (pd.DataFrame): Estimated target values.
        sample_weight (array-like of shape, optional): Sample weights. Defaults to None.
        multioutput (str, optional): Defines aggregating of multiple output values. Defaults to "uniform_average".
        squared (bool, optional): If True returns MSE value, if False returns RMSE value. Defaults to True.

    Returns:
        float: mean_squared_error
    """
    return sklearn.metrics.mean_squared_error(y_true, y_pred, sample_weight=sample_weight, multioutput=multioutput,
                                              squared=squared)


def mean_squared_log_error(y_true, y_pred, *, sample_weight=None, multioutput="uniform_average", squared=True):
    """Implements mean_squared_log_error

    Args:
        y_true (pd.DataFrame): Ground truth (correct) target values.
        y_pred (pd.DataFrame): Estimated target values.
        sample_weight (array-like of shape, optional): Sample weights. Defaults to None.
        multioutput (str, optional): Defines aggregating of multiple output values. Defaults to "uniform_average".
        squared (bool, optional): If True returns MSLE (mean squared log error) value. If False returns RMSLE (root mean squared log error) value.

    Returns:
        float: mean squared log error
    """
    return sklearn.metrics.mean_squared_log_error(y_true, y_pred, sample_weight=sample_weight, multioutput=multioutput,
                                                  squared=squared)


def median_absolute_error(y_true, y_pred, *, multioutput="uniform_average", sample_weight=None):
    """Implements median_absolute_error

    Args:
        y_true (pd.DataFrame): Ground truth (correct) target values.
        y_pred (pd.DataFrame): Estimated target values.
        sample_weight (array-like of shape, optional): Sample weights. Defaults to None.
        multioutput (str, optional): Defines aggregating of multiple output values. Defaults to "uniform_average".

    Returns:
        float: median absolute error
    """
    return sklearn.metrics.median_absolute_error(y_true, y_pred, multioutput=multioutput, sample_weight=sample_weight)


def r2_score(y_true, y_pred, *, sample_weight=None, multioutput="uniform_average", force_finite=True):
    """Implements r2_score metric

    Args:
        y_true (pd.DataFrame): Ground truth (correct) target values.
        y_pred (pd.DataFrame): Estimated target values.
        sample_weight (array-like of shape, optional): Sample weights. Defaults to None.
        multioutput (str, optional): Defines aggregating of multiple output values. Defaults to "uniform_average".
        force_finite (bool, optional): flag indicating if NaN and -Inf scores resulting from constant data should be replaced with real numbers. Defaults to True.

    Returns:
        float: r2 score
    """
    return sklearn.metrics.r2_score(y_true, y_pred, sample_weight=sample_weight, multioutput=multioutput,
                                    force_finite=force_finite)


def explained_variance_score(y_true, y_pred, *, sample_weight=None, multioutput="uniform_average", force_finite=True):
    """Implements explained_varience_score metric

    Args:
        y_true (pd.DataFrame): Ground truth (correct) target values.
        y_pred (pd.DataFrame): Estimated target values.
        sample_weight (array-like of shape, optional): Sample weights. Defaults to None.
        multioutput (str, optional): Defines aggregating of multiple output values. Defaults to "uniform_average".
        force_finite (bool, optional): flag indicating if NaN and -Inf scores resulting from constant data should be replaced with real numbers. Defaults to True.

    Returns:
        float: explained varience score
    """
    return sklearn.metrics.explained_variance_score(y_true, y_pred, sample_weight=sample_weight,
                                                    multioutput=multioutput, force_finite=force_finite)


def mean_tweedie_deviance(y_true, y_pred, *, sample_weight=None, power=0):
    """Implements mean_tweedie_deviance metric

    Args:
        y_true (pd.DataFrame): Ground truth (correct) target values.
        y_pred (pd.DataFrame): Estimated target values.
        sample_weight (array-like of shape, optional): Sample weights. Defaults to None.
        power (int, optional): Tweedie power parameter. Defaults to 0.

    Returns:
        float: mean tweedie deviance
    """
    return sklearn.metrics.mean_tweedie_deviance(y_true, y_pred, sample_weight=sample_weight, power=power)


def mean_poisson_deviance(y_true, y_pred, *, sample_weight=None):
    """Implements mean_poisson_deviance.

    Args:
        y_true (pd.DataFrame): Ground truth (correct) target values.
        y_pred (pd.DataFrame): Estimated target values.
        sample_weight (array-like of shape, optional): Sample weights. Defaults to None.

    Returns:
        float: mean poisson deviance
    """
    return sklearn.metrics.mean_poisson_deviance(y_true, y_pred, sample_weight=sample_weight)


def mean_gamma_deviance(y_true, y_pred, *, sample_weight=None):
    """Implements mean_gamma_deviance

    Args:
        y_true (pd.DataFrame): Ground truth (correct) target values.
        y_pred (pd.DataFrame): Estimated target values.
        sample_weight (array-like of shape, optional): Sample weights. Defaults to None.

    Returns:
        float: mean gamma deviance
    """
    return sklearn.metrics.mean_gamma_deviance(y_true, y_pred, sample_weight=sample_weight)
