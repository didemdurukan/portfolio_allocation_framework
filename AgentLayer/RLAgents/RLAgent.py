from abc import ABC, abstractmethod
from AgentLayer.Agent import Agent
class RLAgent(Agent, ABC):

    @abstractmethod
    def train_model():
        pass

    @abstractmethod
    def predict():
        
        pass

    @abstractmethod
    def save_model():
        pass

    @abstractmethod
    def load_model():
        pass