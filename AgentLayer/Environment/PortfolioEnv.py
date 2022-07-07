from abc import ABC, abstractmethod

from gym.utils import seeding
from stable_baselines3 import A2C
from stable_baselines3.common.vec_env import DummyVecEnv
from stable_baselines3 import A2C as sb_A2C
from gym import spaces
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import yaml
import gym
from datetime import datetime
from AgentLayer.Environment.Environment import Environment


class PortfolioEnv(Environment):
    """Provides methods for generating Portfolio Environment

    Attributes
    ----------        
        df: pd.DataFrame
            input data
        stock_dim: int
            number of unique securities in the investment universe
        hmax: float
            maximum number of shares to trade
        initial_amount: float
            initial cash value
        transaction_cost_pct: float
            transaction cost percentage per trade
        reward_scaling: float
            scaling factor for reward as training progresses
        state_space: gym.spaces.Box object  
            state space
        action_space: gym.spaces.Box object  
            action space
        feature_list: list
            a list of features to be used for the state observations
        lookback: int
            lookback value
        day: int
            an increment number to control date
        observation space: gym.spaces.Box object 
            observation space
        data: pd.DataFrame
            data to be used.
        covs: pd.Series
            covarience values from the data
        state: np.array
            states
        terminal: bool
            if terminal state
        portfolio_value:
            portfolio value
        asset_memory: pd.DataFrame
            used for memorizing portfolio value at each step
        portfolio_return_memory: pd.DataFrame
            used for memorizing return at each step
        actions_memory: pd.DataFrame
            actions memory
        date_memory: pd.DataFrame
            date memory


    Methods
    -------
        step()
            steps the environment with the given action
        reset()
            reset the environment
        render()
            gym environment rendering
        save_asset_memory()
            return account value at each time step
        save_action_memory()
            return actions/positions at each time step
        get_env()
            generates environment.

    """

    def __init__(self,
                 df: pd.DataFrame,  # input data
                 stock_dim: int,  # number of unique securities in the investment universe
                 hmax: float,  # maximum number of shares to trade
                 initial_amount: float,  # initial cash value
                 transaction_cost_pct: float,  # transaction cost percentage per trade
                 reward_scaling: float,  # scaling factor for reward as training progresses
                 # the dimension of input features (state space)
                 state_space: int,
                 action_space: int,  # number of actions, which is equal to portfolio dimension
                 feature_list: list,  # a list of features to be used as observations
                 lookback=252,  #
                 day=0):  # an increment number to control date
        """Initializer for Portfolio Envrionment object

        Args:
            df: pd.DataFrame
                input data
            stock_dim: int
                number of unique securities in the investment universe
            hmax: float
                maximum number of shares to trade
            initial_amount: float
                initial cash value
            transaction_cost_pct: float
                transaction cost percentage per trade
            reward_scaling: float
                scaling factor for reward as training progresses
            state_space: int  
                the dimension of input features (state space)
            action_space: int
                number of actions, which is equal to portfolio dimension
            feature_list: list
                a list of features to be used for the state observations
            lookback: int
                lookback value
            day: int
                an increment number to control date

        """

        self.df = df
        self.day = day
        self.lookback = lookback
        self.stock_dim = stock_dim
        self.hmax = hmax
        self.initial_amount = initial_amount
        self.transaction_cost_pct = transaction_cost_pct
        self.reward_scaling = reward_scaling
        self.state_space = state_space
        self.action_space = action_space
        self.feature_list = feature_list

        # action_space normalization and shape is self.stock_dim
        self.action_space = spaces.Box(
            low=0, high=1, shape=(self.action_space,))
        self.observation_space = spaces.Box(low=-np.inf, high=np.inf,
                                            shape=(self.state_space + len(self.feature_list), self.state_space))

        # load data from a pandas dataframe

        self.data = self.df.loc[self.day, :]
        self.covs = self.data['cov_list'].values[0]
        self.state = np.append(np.array(self.covs), [
                               self.data[feat].values.tolist() for feat in self.feature_list], axis=0)
        self.terminal = False
        #self.turbulence_threshold = turbulence_threshold
        # initalize state: initial portfolio return + individual stock return + individual weights
        self.portfolio_value = self.initial_amount

        # memorize portfolio value each step
        self.asset_memory = [self.initial_amount]
        # memorize portfolio return each step
        self.portfolio_return_memory = [0]
        self.actions_memory = [[1 / self.stock_dim] * self.stock_dim]
        self.date_memory = [self.data.date.unique()[0]]

    def reset(self):
        """Resets the envrionment

        Returns:
            np.array : states
        """
        self.asset_memory = [self.initial_amount]
        self.day = 0
        self.data = self.df.loc[self.day, :]
        # load states
        self.covs = self.data['cov_list'].values[0]
        self.state = np.append(np.array(self.covs), [
                               self.data[feat].values.tolist() for feat in self.feature_list], axis=0)

        self.portfolio_value = self.initial_amount
        self.terminal = False
        self.portfolio_return_memory = [0]
        self.actions_memory = [[1/self.stock_dim] * self.stock_dim]
        self.date_memory = [self.data.date.unique()[0]]
        return self.state

    def step(self, actions):
        """Steps the environment with the given action.
        Args:
            actions ([int] or [float]) : the action

        Returns:
            np.array : state
            int : reward -> new portfolio value or end portfolio value
            bool : if terminal state

        """
        self.terminal = self.day >= len(self.df.index.unique()) - 1
        if self.terminal:
            df = pd.DataFrame(self.portfolio_return_memory)
            df.columns = ['daily_return']
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
            transaction_fee = self.transaction_cost_pct * self.asset_memory[-1] * sum(
                [abs(a_i - b_i) for a_i, b_i in zip(self.actions_memory[-1], self.actions_memory[-2])])  # transaction_fee
            last_day_memory = self.data
            # load next state
            self.day += 1
            self.data = self.df.loc[self.day, :]
            self.covs = self.data['cov_list'].values[0]
            self.state = np.append(np.array(self.covs), [
                                   self.data[feat].values.tolist() for feat in self.feature_list], axis=0)
            portfolio_return = sum(
                ((self.data.close.values / last_day_memory.close.values) - 1) * weights)
            # update portfolio value
            new_portfolio_value = self.portfolio_value * \
                (1 + portfolio_return) - transaction_fee  # transaction_fee
            self.portfolio_value = new_portfolio_value

            # save into memory
            self.portfolio_return_memory.append(portfolio_return)
            self.date_memory.append(self.data["date"].unique()[0])
            self.asset_memory.append(new_portfolio_value)

            # the reward is the new portfolio value or end portfolio value
            self.reward = new_portfolio_value

        return self.state, self.reward, self.terminal, {}

    def render(self, mode='human'):
        """Gym environment rendering

        Args:
            mode (str, optional): the rendering type. Defaults to 'human'.

        Returns:
            np.array: state
        """
        return self.state

    def save_asset_memory(self):
        """returns account value at each time step

        Returns:
            pd.DataFrame : account value
        """
        date_list = self.date_memory
        portfolio_return = self.portfolio_return_memory
        df_account_value = pd.DataFrame(
            {'date': date_list, 'daily_return': portfolio_return})
        return df_account_value

    def save_action_memory(self):
        """Returns actions/positions at each time step

        Returns:
            pd.DataFrame : actions
        """
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

    def get_env(self):
        """Creates the envrionment

        Returns:
            Vectorized Environment : environment
            np.array : array of observations
        """
        e = DummyVecEnv([lambda: self])
        obs = e.reset()
        return e, obs
