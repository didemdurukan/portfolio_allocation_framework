from abc import ABC, abstractmethod
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
from AgentLayer.RLAgents.RLAgent import RLAgent
class A2C(RLAgent):

    def __init__(self,
                 policy= "MlpPolicy",
                 env= None,
                 learning_rate: float = 7e-4,
                 n_steps: int = 5,
                 gamma: float = 0.99,
                 gae_lambda: float = 1.0,
                 ent_coef: float = 0.0,
                 vf_coef: float = 0.5,
                 max_grad_norm: float = 0.5,
                 rms_prop_eps: float = 1e-5,
                 use_rms_prop: bool = True,
                 use_sde: bool = False,
                 sde_sample_freq: int = -1,
                 normalize_advantage: bool = False,
                 tensorboard_log=None,
                 create_eval_env: bool = False,
                 policy_kwargs=None,
                 verbose: int = 0,
                 seed=None,
                 device="auto",
                 _init_setup_model: bool = True):

        self.env = env
        # self.model = A2C(model_params["policy"], model_params["environment"], model_params["verbose"])
        self.model = sb_A2C(policy = policy,
                            env=self.env,
                            learning_rate = learning_rate,
                            n_steps = n_steps,
                            gamma = gamma,
                            gae_lambda= gae_lambda,
                            ent_coef = ent_coef,
                            vf_coef = vf_coef,
                            max_grad_norm = max_grad_norm,
                            rms_prop_eps= rms_prop_eps,
                            use_rms_prop= use_rms_prop,
                            use_sde= use_sde,
                            sde_sample_freq= sde_sample_freq,
                            normalize_advantage= normalize_advantage,
                            tensorboard_log=tensorboard_log,  
                            create_eval_env= create_eval_env,
                            policy_kwargs=policy_kwargs,
                            verbose=verbose,
                            seed=seed,
                            device= device,
                            _init_setup_model = _init_setup_model)

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

    def save_model(self, path):
        self.model.save(path)

    def load_model(self, path):
        self.model = self.model.load(path)
        return self.model