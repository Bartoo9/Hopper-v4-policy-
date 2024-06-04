from stable_baselines3.common.evaluation import evaluate_policy
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import VecNormalize, SubprocVecEnv
from stable_baselines3.common.env_util import make_vec_env
from train import ChangeMassWrapper
import matplotlib.pyplot as plt
import numpy as np

TF_ENABLE_ONEDNN_OPTS=0

torso_masses = list(range(3,10))

results = {'3': {mass: [] for mass in torso_masses}}
std_results = {'3': {mass: [] for mass in torso_masses}}

if __name__ == '__main__':



    model = PPO.load(r"C:\Users\Barto\Desktop\Deep Reinforcement Learning\assignment1\models_PPO_ensemble\PPO_hopper.zip")

    for mass in torso_masses:
        env = make_vec_env('Hopper-v4', n_envs=6, seed=2, vec_env_cls=SubprocVecEnv,
                            wrapper_class=ChangeMassWrapper, wrapper_kwargs={'torso_mass': mass})
        
        env = VecNormalize.load(r'C:\Users\Barto\Desktop\Deep Reinforcement Learning\assignment1\models_PPO_ensemble\PPO_normalize.pkl', env)
        env.training = False
        env.norm_reward = False


        mean_reward, std_reward = evaluate_policy(model, env, n_eval_episodes=1000)
        results[f'{3}'][mass].append(mean_reward)
        std_results[f'{3}'][mass].append(std_reward)

        print(results)
        print(std_results)

        env.close()


        mean_results = {k: [np.mean(v) for v in results[k].values()] for k in results.keys()}
        mean_std_results = {k: [np.mean(v) for v in std_results[k].values()] for k in std_results.keys()}
        print(mean_results)
        print(mean_std_results)

    plt.figure(figsize=(8, 8))
    plt.errorbar(torso_masses, mean_results[f'{3}'],
                yerr=mean_std_results[f'{3}'], label=f'{3}kg model', alpha=0.5,
                fmt='o', color ='black', ecolor='lightgray', elinewidth=3, capsize=0)
    plt.plot(torso_masses, mean_results[f'{3}'], 'b-')
    plt.xlabel('Torso Masses')
    plt.ylabel('Mean Rewards')
    plt.title('Generalization of Hopper Model')
    plt.ylim(0, 4000)
    plt.show()