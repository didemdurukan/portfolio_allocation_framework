from abc import ABC, abstractmethod
class Agent(ABC):

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