from abc import ABC, abstractmethod
from stable_baselines3.common.vec_env import DummyVecEnv
from stable_baselines3 import DDPG as sb_DDPG
from stable_baselines3.common.noise import ActionNoise
from stable_baselines3.common.buffers import ReplayBuffer
from stable_baselines3.common.type_aliases import Schedule
from AgentLayer.RLAgents.RLAgent import RLAgent
from typing import Any, Dict, Optional, Tuple, Type, Union
import torch as th


class DDPG(RLAgent):
    """Provides methods for DDPG Agent.

    Attributes
    ----------        
        env: DummyVecEnv
            The environment to learn from 
        model: stable_baselines3.DDPG Agent
            RL Agent

    Methods
    -------
        train_model()
            trains the agent.
        predict()
            prediction method.
        save_model()
            saves the model.
        load_model()
            loads the model.
    """

    def __init__(self,
                 policy="MlpPolicy",
                 env=None,
                 learning_rate=1e-3,
                 buffer_size=1_000_000,  # 1e6
                 learning_starts=100,
                 batch_size=100,
                 tau=0.005,
                 gamma=0.99,
                 train_freq=1,
                 gradient_steps: int = -1,
                 action_noise: Optional[ActionNoise] = None,
                 replay_buffer_class: Optional[ReplayBuffer] = None,
                 replay_buffer_kwargs: Optional[Dict[str, Any]] = None,
                 optimize_memory_usage: bool = False,
                 tensorboard_log: Optional[str] = None,
                 create_eval_env: bool = False,
                 policy_kwargs: Optional[Dict[str, Any]] = None,
                 verbose: int = 0,
                 seed: Optional[int] = None,
                 device: Union[th.device, str] = "auto",
                 _init_setup_model: bool = True):
        """Initializer for DDPG object.

        Args:
            policy (str, optional): The policy to use. Defaults to "MlpPolicy".
            env (DummyVecEnv, optional): The environment to learn from . Defaults to None.
            learning_rate (float, optional): learning rate for adam optimizer. Defaults to 1e-3.
            buffer_size (int, optional): size of the replay buffer. Defaults to 1_000_000.
            batch_size (int, optional): Minibatch size for each gradient update. Defaults to 100.
            tau (float, optional): the soft update coefficient . Defaults to 0.005.
            gamma (float, optional): the discount factor. Defaults to 0.99.
            train_freq (int, optional): Update the model every train_freq steps. Defaults to 1.
            gradient_steps (int, optional): How many gradient steps to do after each rollout. Defaults to -1.
            action_noise (Optional[ActionNoise], optional): the action noise type. Defaults to None.
            replay_buffer_class (Optional[ReplayBuffer], optional): Replay buffer class to use. Defaults to None.
            replay_buffer_kwargs (Optional[Dict[str, Any]], optional): Keyword arguments to pass to the replay buffer on creation. Defaults to None.
            optimize_memory_usage (bool, optional): Enable a memory efficient variant of the replay buffer at a cost of more complexity. Defaults to False.
            tensorboard_log (Optional[str], optional): the log location for tensorboard. Defaults to None. 
            create_eval_env (bool, optional): Whether to create a second environment that will be used for evaluating the agent periodically. Defaults to False.
            policy_kwargs (Optional[Dict[str, Any]], optional): additional arguments to be passed to the policy on creation. Defaults to None.
            verbose (int, optional): the verbosity level: 0 no output, 1 info, 2 debug. Defaults to 0.
            seed (Optional[int], optional): Seed for the pseudo random generators. Defaults to None.
            device (Union[th.device, str], optional):  on which the code should be run. Defaults to "auto".
            _init_setup_model (bool, optional): Whether or not to build the network at the creation of the instance. Defaults to True.
        """

        self.env = env

        self.model = sb_DDPG(policy=policy,
                             env=self.env,
                             learning_rate=learning_rate,
                             buffer_size=buffer_size,
                             learning_starts=learning_starts,
                             batch_size=batch_size,
                             tau=tau,
                             gamma=gamma,
                             train_freq=train_freq,
                             gradient_steps=gradient_steps,
                             action_noise=action_noise,
                             replay_buffer_class=replay_buffer_class,
                             replay_buffer_kwargs=replay_buffer_kwargs,
                             optimize_memory_usage=optimize_memory_usage,
                             tensorboard_log=tensorboard_log,
                             create_eval_env=create_eval_env,
                             policy_kwargs=policy_kwargs,
                             verbose=verbose,
                             seed=seed,
                             device=device,
                             _init_setup_model=_init_setup_model)

    def train_model(self, **train_params):
        """Trains the model

        Args:
            train_params (dict) : train parameters

        Returns:
            model: trained model.
        """
        self.model = self.model.learn(**train_params)
        return self.model

    def predict(self, environment, **test_params):
        """Does the prediction

        Args:
            environment (DummyVecEnv): test environment

        Returns:
            pd.DataFrame: portfolio
            ndarray : actions memory
        """

        env_test, obs_test = environment.get_env()
        account_memory = []
        actions_memory = []

        env_test.reset()
        for i in range(len(environment.df.index.unique())):
            action, _states = self.model.predict(obs_test, **test_params)
            obs_test, rewards, dones, info = env_test.step(action)
            if i == (len(environment.df.index.unique()) - 2):
                account_memory = env_test.env_method(
                    method_name="save_asset_memory")
                actions_memory = env_test.env_method(
                    method_name="save_action_memory")
            if dones[0]:
                print("hit end!")
                break

        portfolio_df = account_memory[0]
        portfolio_df = portfolio_df.rename(
            columns={"daily_return": "account_value"})
        portfolio_df.iloc[0, portfolio_df.columns.get_loc(
            "account_value")] = environment.initial_amount
        values = list(portfolio_df["account_value"])
        for i in range(1, len(values)):
            values[i] = (values[i] + 1) * values[i-1]

        portfolio_df["account_value"] = values
        return portfolio_df, actions_memory[0]

    def load_model(self, path):
        """Loads the model

        Args:
            path (str): path from loading the model.

        Returns:
            model: loaded model
        """
        self.model = self.model.load(path)
        return self.model

    def save_model(self, path):
        """Saves the model

        Args:
            path (str): path for where to save the model.
        """
        self.model.save(path)
