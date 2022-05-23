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
        self.model = self.model.learn(**train_params)
        return self.model

    def predict(self, environment, **test_params):
        env_test, obs_test = environment.get_env()
        """make a prediction"""
        account_memory = []
        actions_memory = []

        env_test.reset()
        for i in range(len(environment.df.index.unique())):
            action, _states = self.model.predict(obs_test, **test_params)
            obs_test, rewards, dones, info = env_test.step(action)
            if i == (len(environment.df.index.unique()) - 2):
                account_memory = env_test.env_method(method_name="save_asset_memory")
                actions_memory = env_test.env_method(method_name="save_action_memory")
            if dones[0]:
                print("hit end!")
                break

        return account_memory[0], actions_memory[0]

    def load_model(self, path):
        self.model = self.model.load(path)
        return self.model

    def save_model(self, path):
        self.model.save(path)
