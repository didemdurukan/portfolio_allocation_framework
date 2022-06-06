from abc import ABC, abstractmethod
from stable_baselines3 import A2C
from stable_baselines3.common.vec_env import DummyVecEnv
from stable_baselines3 import A2C as sb_A2C
from stable_baselines3 import PPO as sb_PPO
from gym import spaces
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import yaml
import gym
from AgentLayer.RLAgents.RLAgent import RLAgent
from stable_baselines3.common.buffers import ReplayBuffer
from stable_baselines3.common.type_aliases import Schedule
from typing import Any, Dict, Optional, Tuple, Type, Union
import torch as th


class PPO(RLAgent):
    """"Provides methods for PPO Agent.

    Attributes
    ----------        
        env: DummyVecEnv
            The environment to learn from 
        model: stable_baselines3.PPO Agent
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
                 learning_rate=3e-4,
                 n_steps: int = 2048,
                 batch_size: int = 64,
                 n_epochs: int = 10,
                 gamma: float = 0.99,
                 gae_lambda: float = 0.95,
                 clip_range: Union[float, Schedule] = 0.2,
                 clip_range_vf: Union[None, float, Schedule] = None,
                 normalize_advantage: bool = True,
                 ent_coef: float = 0.0,
                 vf_coef: float = 0.5,
                 max_grad_norm: float = 0.5,
                 use_sde: bool = False,
                 sde_sample_freq: int = -1,
                 target_kl: Optional[float] = None,
                 tensorboard_log: Optional[str] = None,
                 create_eval_env: bool = False,
                 policy_kwargs: Optional[Dict[str, Any]] = None,
                 verbose: int = 0,
                 seed: Optional[int] = None,
                 device: Union[th.device, str] = "auto",
                 _init_setup_model: bool = True):
        """Initializer for PPO Object"

        Args:
            policy (str, optional): The policy to use. Defaults to "MlpPolicy".
            env (DummyVecEnv, optional): The environment to learn from. Defaults to None.
            learning_rate (float, optional): The learning rate. Defaults to 3e-4.
            n_steps (int, optional): The number of steps to run for each environment per update 
            (i.e. batch size is n_steps * n_env where n_env is number of environment copies running in parallel)
            . Defaults to 2048.
            batch_size (int, optional): Minibatch size. Defaults to 64.
            n_epochs (int, optional): number of epochs when optimizing the surrogate loss. Defaults to 10.
            gamma (float, optional): discount factor. Defaults to 0.99.
            gae_lambda (float, optional): Factor for trade-off of bias vs variance for Generalized Advantage Estimator Equivalent to classic advantage when set to 1. Defaults to 0.95.
            clip_range (Union[float, Schedule], optional): Clipping parameter. Defaults to 0.2.
            clip_range_vf (Union[None, float, Schedule], optional): Clipping parameter for the value function. Defaults to None.
            normalize_advantage (bool, optional): Whether to normalize or not the advantage. Defaults to True.
            ent_coef (float, optional): Entropy coefficient for the loss calculation. Defaults to 0.0.
            vf_coef (float, optional): Value function coefficient for the loss calculation. Defaults to 0.5.
            max_grad_norm (float, optional): The maximum value for the gradient clipping. Defaults to 0.5.
            use_sde (bool, optional): Whether to use generalized State Dependent Exploration (gSDE) instead of action noise exploration. Defaults to False.
            sde_sample_freq (int, optional): Sample a new noise matrix every n steps when using gSDE. Defaults to -1.
            target_kl (Optional[float], optional): Limit the KL divergence between updates. Defaults to None.
            tensorboard_log (Optional[str], optional): the log location for tensorboard. Defaults to None.
            create_eval_env (bool, optional): Whether to create a second environment that will be used for evaluating the agent periodically. Defaults to False.
            policy_kwargs (Optional[Dict[str, Any]], optional): additional arguments to be passed to the policy on creation. Defaults to None.
            verbose (int, optional): the verbosity level: 0 no output, 1 info, 2 debug. Defaults to 0.
            seed (Optional[int], optional): Seed for the pseudo random generators. Defaults to None.
            device (Union[th.device, str], optional): Device (cpu, cuda, …) on which the code should be run. Defaults to "auto".
            _init_setup_model (bool, optional): Whether or not to build the network at the creation of the instance. Defaults to True.
        """

        self.env = env
        self.model = sb_PPO(policy=policy,
                            env=self.env,
                            learning_rate=learning_rate,
                            n_steps=n_steps,
                            gamma=gamma,
                            batch_size=batch_size,
                            n_epochs=n_epochs,
                            gae_lambda=gae_lambda,
                            clip_range=clip_range,
                            clip_range_vf=clip_range_vf,
                            normalize_advantage=normalize_advantage,
                            ent_coef=ent_coef,
                            vf_coef=vf_coef,
                            max_grad_norm=max_grad_norm,
                            use_sde=use_sde,
                            sde_sample_freq=sde_sample_freq,
                            target_kl=target_kl,
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
            model: trained model
        """
        self.model = self.model.learn(**train_params)
        return self.model

    def predict(self, environment, **test_params):
        """Does the prediction

        Args:
            environment (DummyVecEnv): test environment
            test_params (dict) : test parameters

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
