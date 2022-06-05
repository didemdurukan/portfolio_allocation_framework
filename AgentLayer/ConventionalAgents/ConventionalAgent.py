from abc import ABC, abstractmethod
from AgentLayer.Agent import Agent


class ConventionalAgent(Agent, ABC):
    """Defines the base structure of Conventional Agents classes.

    Attributes
    ----------
        -

    Methods
    -------
        train_model()
            abstract method
        predict()
            abstract method
        save_model()
            abstract method
        load_model()
            abstract method
        _return_predict()
            abstract method
        _weight_optimization()
            abstract method

    """

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
