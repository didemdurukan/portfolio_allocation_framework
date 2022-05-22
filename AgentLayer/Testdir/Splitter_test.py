import numpy as np
from AgentLayer.DataSplitter.datasplitter import ExpandingWindowSplitter, BlockingTimeSeriesSplitter

X = np.random.randn(15, 2)
y = np.random.randint(0, 2, 15)
tscv = ExpandingWindowSplitter(n_splits=3, test_size=2, gap=1)
print(tscv)
for train_index, test_index in tscv.split(X=X):
    print("TRAIN:", train_index, "TEST:", test_index)
    X_train, X_test = X[train_index], X[test_index]
    y_train, y_test = y[train_index], y[test_index]

print("*****************************************")
blocked_tscv = BlockingTimeSeriesSplitter(n_splits=3, test_size=2, gap=0)
for train_index, test_index in blocked_tscv.split(X=X):
    print("TRAIN:", train_index, "TEST:", test_index)
    X_train, X_test = X[train_index], X[test_index]
    y_train, y_test = y[train_index], y[test_index]