from abc import ABC, abstractmethod
import numpy as np
import gym


class Environment(gym.Env, ABC):

    @abstractmethod
    def reset(self):
        pass

    @abstractmethod
    def step(self, action):
        pass

    @abstractmethod
    def render(self, mode="human"):
        pass

    @abstractmethod
    def get_env(self):
        pass

    @staticmethod
    def softmax_normalization(actions):
        numerator = np.exp(actions)
        denominator = np.sum(np.exp(actions))
        softmax_output = numerator / denominator
        return softmax_output
