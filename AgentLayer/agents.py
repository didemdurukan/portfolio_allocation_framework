from abc import ABC, abstractmethod
from stable_baselines3 import A2C as sb_A2C

# TODO: adhere to PyPI structure
class Agent(ABC):

    @abstractmethod
    def train_model(self):
        pass

    @abstractmethod
    def predict(self):
        pass

    @abstractmethod
    def save_model(self, path):
        pass

    @abstractmethod
    def load_model(self, path):
        pass


class ConventionalAgent(Agent):

    @abstractmethod
    def train_model(self):
        pass

    @abstractmethod
    def predict(self):
        pass

    @abstractmethod
    def save_model(self, path):
        pass

    @abstractmethod
    def load_model(self, path):
        pass

    @abstractmethod
    def _return_predict(self):
        pass

    @abstractmethod
    def _weight_optimization(self):
        pass


class RLAgent(Agent):

    @abstractmethod
    def train_model(self,
                    total_timesteps,
                    callback=None,
                    log_interval=100,
                    tb_log_name="A2C",
                    reset_num_timesteps=True):
        pass

    @abstractmethod
    def predict(self):
        pass

    @abstractmethod
    def save_model(self, path):
        pass

    @abstractmethod
    def load_model(self, path):
        pass


class A2C(RLAgent, sb_A2C):

    def __init__(self,
                 policy,
                 env,
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

        super(A2C, self).__init__(policy=policy,
                                  env=env,
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

    def train_model(self,
                    total_timesteps,
                    callback=None,
                    log_interval=100,
                    tb_log_name="A2C",
                    reset_num_timesteps=True):
        return super().learn(total_timesteps, callback, log_interval, tb_log_name, reset_num_timesteps)

    def predict(self, **test_params):

        test_env, test_obs = test_params["environment"].environment()
        """make a prediction"""
        account_memory = []
        actions_memory = []

        test_env.reset()
        for i in range(len(test_params["environment"].df.index.unique())):
            action, _states = self.model.predict(test_obs, deterministic=test_params["deterministic"])
            test_obs, rewards, dones, info = test_env.step(action)
            if i == (len(test_params["environment"].df.index.unique()) - 2):
                account_memory = test_env.env_method(method_name="save_asset_memory")
                actions_memory = test_env.env_method(method_name="save_action_memory")
            if dones[0]:
                print("hit end!")
                break

        return account_memory[0], actions_memory[0]

    def save_model(self, path):
        self.model.save(path)

    def load_model(self, path):
        self.model = self.model.load(path)
        return self.model
