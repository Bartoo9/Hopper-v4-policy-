import gymnasium as gym
from stable_baselines3.common.vec_env import VecNormalize, SubprocVecEnv, DummyVecEnv
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3 import PPO
from stable_baselines3.common.utils import set_random_seed
import numpy as np
import matplotlib.pyplot as plt
import torch.nn as nn
from stable_baselines3.common.callbacks import EvalCallback, StopTrainingOnRewardThreshold

class ChangeMassWrapper(gym.Wrapper):
    def __init__(self, env, min = 3, max=9):
        super().__init__(env)
        self.min = min
        self.max = max
    
    def reset(self, **kwargs):
        random_mass = np.random.uniform(self.min, self.max)
        self.env.model.body_mass[1] = random_mass
        return self.env.reset(**kwargs)

if __name__ == '__main__':
    seed = 123
    
    env = make_vec_env('Hopper-v4', n_envs=6, seed=seed, vec_env_cls=SubprocVecEnv,
                        wrapper_class=ChangeMassWrapper, wrapper_kwargs={'min': 3, 'max': 9})
    env = VecNormalize(env, norm_obs=True, norm_reward=True)

    modeldir2 = 'models_PPO_ensemble'
    logdir2 = 'logs_PPO_ensemble'
    eval_callback = EvalCallback(env, eval_freq=10000, verbose= 1,
                                deterministic=True, render=False)

    model = PPO('MlpPolicy', env=env, verbose=1, device='cuda', tensorboard_log=logdir2, seed=seed) 

    TIMESTEPS = 4000000

    model.learn(total_timesteps=TIMESTEPS, callback = eval_callback)

    model.save(f"{modeldir2}/PPO_hopper.zip")
    env.save(f"{modeldir2}/PPO_normalize.pkl")

    obs = env.reset()
    while True:
        action, _states = model.predict(obs)
        obs, rewards, dones, info = env.step(action)
        env.render('human')