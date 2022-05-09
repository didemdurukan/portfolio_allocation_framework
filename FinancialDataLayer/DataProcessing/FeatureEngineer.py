from abc import ABC, abstractmethod


class FeatureEngineer(ABC):

    @abstractmethod
    def extend_data(self, df):
        pass
