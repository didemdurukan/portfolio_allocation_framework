from abc import ABC, abstractmethod
import gym
from gym.utils import seeding
from gym.vector.utils import spaces
from matplotlib import pyplot as plt
from stable_baselines3.common.vec_env import DummyVecEnv
import numpy as np
import pandas as pd


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


class PortfolioEnv(Environment):

    def __init__(self,
                 df: pd.DataFrame,  # input data
                 port_dim: int,  # number of unique securities in the investment universe
                 hmax: float,  # maximum number of shares to trade
                 initial_amount: float,  # initial cash value
                 transaction_cost_pct: float,  # transaction cost percentage per trade
                 reward_scaling: float,  # scaling factor for reward as training progresses
                 state_space: int,  # the dimension of input features (state space)
                 action_space: int,  # number of actions, which is equal to portfolio dimension
                 tech_indicator_list: list,  # a list of technical indicator names
                 turbulence_threshold=None,  # a threshold to control risk aversion
                 lookback=252,  #
                 day=0):  # an increment number to control date

        self.df = df
        self.day = day
        self.lookback = lookback
        self.port_dim = port_dim
        self.hmax = hmax
        self.initial_amount = initial_amount
        self.transaction_cost_pct = transaction_cost_pct
        self.reward_scaling = reward_scaling
        self.state_space = state_space
        self.action_space = action_space
        self.tech_indicator_list = tech_indicator_list

        # action_space normalization and shape is self.stock_dim
        self.action_space = spaces.Box(low=0, high=1, shape=(self.action_space,))
        self.observation_space = spaces.Box(low=-np.inf, high=np.inf,
                                            shape=(self.state_space + len(self.tech_indicator_list), self.state_space))

        # print(self.df.head(3))
        # load data from a pandas dataframe
        self.data = self.df.loc[self.day, :]
        #print(self.data.head(3))
        self.covs = self.data['cov_list'][0]
        self.state = np.append(np.array(self.covs), [self.data[tech].tolist() for tech in self.tech_indicator_list],
                               axis=0)
        self.terminal = False
        #self.turbulence_threshold = turbulence_threshold
        # initalize state: initial portfolio return + individual stock return + individual weights
        self.portfolio_value = self.initial_amount

        # memorize portfolio value each step
        self.asset_memory = [self.initial_amount]
        # memorize portfolio return each step
        self.portfolio_return_memory = [0]
        self.actions_memory = [[1 / self.port_dim] * self.port_dim]
        # print(self.data.head(3))
        #self.date_memory = [self.data.date.unique()[0]] # TODO: Solve this issue

    def reset(self):
        pass

    def step(self, actions):
        self.terminal = self.day >= len(self.df.index.unique()) - 1

        if self.terminal:
            df = pd.DataFrame(self.portfolio_return_memory)
            df.columns = ['daily_return']
            plt.plot(df.daily_return.cumsum(), 'r')
            plt.savefig('results/cumulative_reward.png')
            plt.close()

            plt.plot(self.portfolio_return_memory, 'r')
            plt.savefig('results/rewards.png')
            plt.close()

            print("=================================")
            print("begin_total_asset:{}".format(self.asset_memory[0]))
            print("end_total_asset:{}".format(self.portfolio_value))

            df_daily_return = pd.DataFrame(self.portfolio_return_memory)
            df_daily_return.columns = ['daily_return']
            if df_daily_return['daily_return'].std() != 0:
                sharpe = (252 ** 0.5) * df_daily_return['daily_return'].mean() / \
                         df_daily_return['daily_return'].std()
                print("Sharpe: ", sharpe)
            print("=================================")

            return self.state, self.reward, self.terminal, {}

        else:
            weights = Environment.softmax_normalization(actions)
            self.actions_memory.append(weights)
            last_day_memory = self.data

            # load next state
            self.day += 1
            self.data = self.df.loc[self.day, :]
            self.covs = self.data['cov_list'][0]
            self.state = np.append(np.array(self.covs), [self.data[tech].tolist() for tech in self.tech_indicator_list],
                                   axis=0)
            portfolio_return = sum(((self.data.close.values / last_day_memory.close.values) - 1) * weights)
            log_portfolio_return = np.log(sum((self.data.close.values / last_day_memory.close.values) * weights))
            # update portfolio value
            new_portfolio_value = self.portfolio_value * (1 + portfolio_return)
            self.portfolio_value = new_portfolio_value

            # save into memory
            self.portfolio_return_memory.append(portfolio_return)
            self.date_memory.append(self.data["date"].unique()[0])
            self.asset_memory.append(new_portfolio_value)

            # the reward is the new portfolio value or end portfolo value
            self.reward = new_portfolio_value

        return self.state, self.reward, self.terminal, {}

    def render(self, mode='human'):
        return self.state

    def save_asset_memory(self):
        date_list = self.date_memory
        portfolio_return = self.portfolio_return_memory
        # print(len(date_list))
        # print(len(asset_list))
        df_account_value = pd.DataFrame({'date': date_list, 'daily_return': portfolio_return})
        return df_account_value

    def save_action_memory(self):
        # date and close price length must match actions length
        date_list = self.date_memory
        df_date = pd.DataFrame(date_list)
        df_date.columns = ['date']

        action_list = self.actions_memory
        df_actions = pd.DataFrame(action_list)
        df_actions.columns = self.data.tic.values
        df_actions.index = df_date.date
        # df_actions = pd.DataFrame({'date':date_list,'actions':action_list})
        return df_actions

    def _seed(self, seed=None):
        self.np_random, seed = seeding.np_random(seed)
        return [seed]

    def get_env(self):
        e = DummyVecEnv([lambda: self])
        obs = e.reset()
        return e, obs
