from abc import ABC, abstractmethod
from AgentLayer.Agent import Agent


class ConventionalAgent(Agent, ABC):

    @abstractmethod
    def train_model(self, *args, **kwargs):
        pass

    @abstractmethod
    def predict(self, *args, **kwargs):
        pass

    @abstractmethod
    def save_model(self, *args, **kwargs):
        pass

    @abstractmethod
    def load_model(self, *args, **kwargs):
        pass

    @abstractmethod
    def _return_predict(self, *args, **kwargs):
        pass

    @abstractmethod
    def _weight_optimization(self, *args, **kwargs):
        pass

    @staticmethod
    def extract_weights(meta_coefficient):
        pass


