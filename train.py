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
    def __init__(self, env, torso_mass=6):
        super().__init__(env)
        self.torso_mass = torso_mass
        self.env.model.body_mass[1] = self.torso_mass

if __name__ == '__main__':
    seed = 123
    torso_mass = 9

    env = make_vec_env('Hopper-v4', n_envs=6, seed=seed, vec_env_cls=SubprocVecEnv,
                        wrapper_class=ChangeMassWrapper, wrapper_kwargs={'torso_mass': torso_mass})
    env = VecNormalize(env, norm_obs=True, norm_reward=True)

    modeldir2 = 'models_PPO'
    logdir2 = 'logs_PPO'
    eval_callback = EvalCallback(env, eval_freq=10000, verbose= 1,
                                deterministic=True, render=False)

    model = PPO('MlpPolicy', env=env, verbose=1, device='cuda', tensorboard_log=logdir2, seed=seed) 

    TIMESTEPS = 2000000

    model.learn(total_timesteps=TIMESTEPS, callback = eval_callback)

    model.save(f"{modeldir2}/PPO_hopper_{torso_mass}kg_seed_{seed}.zip")
    env.save(f"{modeldir2}/PPO_normalize_{torso_mass}kg_seed_{seed}.pkl")

    obs = env.reset()
    while True:
        action, _states = model.predict(obs)
        obs, rewards, dones, info = env.step(action)
        env.render('human')



