from abc import ABC, abstractmethod
from stable_baselines3.common.vec_env import DummyVecEnv
from stable_baselines3 import TD3 as sb_TD3
from stable_baselines3.common.noise import ActionNoise
from stable_baselines3.common.buffers import ReplayBuffer
from stable_baselines3.common.type_aliases import Schedule
from AgentLayer.RLAgents.RLAgent import RLAgent
from typing import Any, Dict, Optional, Tuple, Type, Union
import torch as th


class TD3(RLAgent):

    def __init__(self,
                 policy="MlpPolicy",
                 env=None,
                 learning_rate: float = 1e-3,
                 buffer_size: int = 1_000_000,  # 1e6
                 learning_starts: int = 100,
                 batch_size: int = 100,
                 tau: float = 0.005,
                 gamma: float = 0.99,
                 train_freq: int = 1,
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

        self.env = env

        self.model = sb_TD3(policy=policy,
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
