from abc import ABC, abstractmethod
from stable_baselines3 import A2C as sb_A2C


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


class A2C(RLAgent):

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
                 tensorboard_log=None,
                 create_eval_env: bool = False,
                 policy_kwargs=None,
                 verbose: int = 0,
                 seed=None,
                 device="auto",
                 _init_setup_model: bool = True):

        self.env = env
        # self.model = A2C(model_params["policy"], model_params["environment"], model_params["verbose"])
        self.model = sb_A2C(policy=policy,
                            env=self.env,
                            tensorboard_log=tensorboard_log,
                            verbose=verbose,
                            policy_kwargs=policy_kwargs,
                            seed=seed,
                            **model_kwargs)

    def train_model(self, **train_params):
        self.model = self.model.learn(total_timesteps=train_params["total_timesteps"],
                                      log_interval=train_params["log_interval"])
        return self.model

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
