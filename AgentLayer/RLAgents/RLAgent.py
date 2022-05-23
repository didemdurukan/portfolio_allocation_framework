from abc import ABC, abstractmethod
from AgentLayer.Agent import Agent
import pandas as pd


class RLAgent(Agent, ABC):

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

    @staticmethod
    def extract_weights(rl_actions_list):
        agent_weight_df = {'date': [], 'weights': []}
        for i in range(len(rl_actions_list)):
            date = rl_actions_list.index[i]
            tic_list = list(rl_actions_list.columns)
            weights_list = rl_actions_list.reset_index()[list(rl_actions_list.columns)].iloc[i].values
            weight_dict = {'tic': [], 'weight': []}
            for j in range(len(tic_list)):
                weight_dict['tic'] += [tic_list[j]]
                weight_dict['weight'] += [weights_list[j]]

            agent_weight_df['date'] += [date]
            agent_weight_df['weights'] += [pd.DataFrame(weight_dict)]

        agent_weights = pd.DataFrame(agent_weight_df)
        return agent_weights
