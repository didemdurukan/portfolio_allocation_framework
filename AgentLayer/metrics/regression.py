import sklearn as sklearn

'''
Metrics to assess performance on regression task.
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
    return sklearn.metrics.max_error(y_true, y_pred)


def mean_absolute_error(y_true, y_pred, *, sample_weight=None, multioutput="uniform_average"):
    return sklearn.metrics.mean_absolute_error(y_true, y_pred, sample_weight=sample_weight, multioutput=multioutput)


def mean_squared_error(y_true, y_pred, *, sample_weight=None, multioutput="uniform_average", squared=True):
    return sklearn.metrics.mean_squared_error(y_true, y_pred, sample_weight=sample_weight, multioutput=multioutput,
                                              squared=squared)


def mean_squared_log_error(y_true, y_pred, *, sample_weight=None, multioutput="uniform_average", squared=True):
    return sklearn.metrics.mean_squared_log_error(y_true, y_pred, sample_weight=sample_weight, multioutput=multioutput,
                                                  squared=squared)


def median_absolute_error(y_true, y_pred, *, multioutput="uniform_average", sample_weight=None):
    return sklearn.metrics.median_absolute_error(y_true, y_pred, multioutput=multioutput, sample_weight=sample_weight)


def r2_score(y_true, y_pred, *, sample_weight=None, multioutput="uniform_average", force_finite=True):
    return sklearn.metrics.r2_score(y_true, y_pred, sample_weight=sample_weight, multioutput=multioutput,
                                    force_finite=force_finite)


def explained_variance_score(y_true, y_pred, *, sample_weight=None, multioutput="uniform_average", force_finite=True):
    return sklearn.metrics.explained_variance_score(y_true, y_pred, sample_weight=sample_weight,
                                                    multioutput=multioutput, force_finite=force_finite)


def mean_tweedie_deviance(y_true, y_pred, *, sample_weight=None, power=0):
    return sklearn.metrics.mean_tweedie_deviance(y_true, y_pred, sample_weight=sample_weight, power=power)


def mean_poisson_deviance(y_true, y_pred, *, sample_weight=None):
    return sklearn.metrics.mean_poisson_deviance(y_true, y_pred, sample_weight=sample_weight)


def mean_gamma_deviance(y_true, y_pred, *, sample_weight=None):
    return sklearn.metrics.mean_gamma_deviance(y_true, y_pred, sample_weight=sample_weight)
