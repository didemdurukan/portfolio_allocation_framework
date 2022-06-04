from abc import ABC, abstractmethod
from stable_baselines3 import A2C
from stable_baselines3 import A2C as sb_A2C
from gym import spaces
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from AgentLayer.RLAgents.RLAgent import RLAgent
from typing import Any, Dict, Optional, Tuple, Type, Union
from stable_baselines3.common.noise import ActionNoise
from stable_baselines3.common.buffers import ReplayBuffer
from stable_baselines3.common.type_aliases import Schedule
import torch as th


class A2C(RLAgent):

    """Provides methods for A2C Agent.
    Attributes
    ----------        
        policy: str
            The policy model to use
        env: DummyVecEnv
            The environment to learn from 
        learning_rate: float
             The learning rate
        n_steps: int
            The number of steps to run for each environment per update 
            (i.e. batch size is n_steps * n_env where n_env is number of environment copies running in parallel)
        gamma : float
            discount factor
        gae_lambda: float
             Factor for trade-off of bias vs variance for Generalized Advantage Estimator Equivalent to classic advantage when set to 1.
        ent_coef: float  
            Entropy coefficient for the loss calculation
        vf_coef : float
            Value function coefficient for the loss calculationv
        max_grad_norm : float  
            The maximum value for the gradient clipping
        rms_prop_eps : float
            RMSProp epsilon. It stabilizes square root computation in denominator of RMSProp update
        use_rms_prop : boolean
            an increment number to control date
        use_sde  : boolean
            Whether to use generalized State Dependent Exploration (gSDE) instead of action noise exploration (default: False)
        sde_sample_freq  : int  
            Sample a new noise matrix every n steps when using gSDE
        normalize_advantage  : boolean
            Whether to normalize or not the advantage
        tensorboard_log  : str
            the log location for tensorboard
        create_eval_env : boolean
            Whether to create a second environment that will be used for evaluating the agent periodically.
        policy_kwargs: dict
            additional arguments to be passed to the policy on creation
        verbose : int
            the verbosity level: 0 no output, 1 info, 2 debug
        seed : int
             Seed for the pseudo random generators
        device : str
            Device (cpu, cuda, …) on which the code should be run.
        _init_setup_model: boolean
            Whether or not to build the network at the creation of the instance.

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
                 tensorboard_log: Optional[str] = None,
                 create_eval_env: bool = False,
                 policy_kwargs: Optional[Dict[str, Any]] = None,
                 verbose: int = 0,
                 seed: Optional[int] = None,
                 device: Union[th.device, str] = "auto",
                 _init_setup_model: bool = True):

        self.env = env

        self.model = sb_A2C(policy=policy,
                            env=self.env,
                            learning_rate=learning_rate,
                            n_steps=n_steps,
                            gamma=gamma,
                            gae_lambda=gae_lambda,
                            ent_coef=ent_coef,
                            vf_coef=vf_coef,
                            max_grad_norm=max_grad_norm,
                            rms_prop_eps=rms_prop_eps,
                            use_rms_prop=use_rms_prop,
                            use_sde=use_sde,
                            sde_sample_freq=sde_sample_freq,
                            normalize_advantage=normalize_advantage,
                            tensorboard_log=tensorboard_log,
                            create_eval_env=create_eval_env,
                            policy_kwargs=policy_kwargs,
                            verbose=verbose,
                            seed=seed,
                            device=device,
                            _init_setup_model=_init_setup_model)

    def train_model(self, **train_params):
        """Trains the model

        Returns:
            model: trained model.
        """
        self.model = self.model.learn(**train_params)
        return self.model

    def predict(self, environment, **test_params):
        """Does the prediction

        Args:
            environment (env): test environment

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

    def save_model(self, path):
        """Saves the model

        Args:
            path (str): path for where to save the model.
        """
        self.model.save(path)

    def load_model(self, path):
        """Loads the model

        Args:
            path (str): path from loading the model.

        Returns:
            model: loaded model
        """
        self.model = self.model.load(path)
        return self.model
