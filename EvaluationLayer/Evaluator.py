from abc import ABC, abstractmethod


class Evaluator(ABC):

    @abstractmethod
    def __init__(self):
        pass

    @abstractmethod
    def backtest_stats(self):
        pass

    @abstractmethod
    def backtest_plot(self):
        pass
