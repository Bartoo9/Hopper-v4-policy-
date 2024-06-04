import gymnasium as gym 
from stable_baselines3 import PPO, TD3
import os 
import argparse

modeldir = 'models_4'
logdir = 'logs_4'

os.makedirs(modeldir, exist_ok=True)
os.makedirs(logdir, exist_ok=True)

def train(env, sb3_algo):
    match sb3_algo: 
        case 'PPO':
            model = PPO('MlpPolicy', env, verbose= 1, device='cuda', tensorboard_log=logdir, gamma=0.999, learning_rate=3e-4,
                        n_steps=2048, batch_size=64, clip_range=0.18, n_epochs = 10, seed=0, gae_lambda=0.95, ent_coef=0.0) 
        case 'TD3':
            model = TD3('MlpPolicy', env, verbose= 1, device='cuda', tensorboard_log=logdir, gamma=0.98, learning_rate=1e-3, 
                        buffer_size=200000, learning_starts=10000, gradient_steps=1, train_freq = 1, policy_kwargs= dict(net_arch=[400, 300]) ,seed=0)
        case _:
            print('No alg')
            return 
    
    TIMESTEPS = 250000
    iters = 0 

    while iters < 4:
        iters += 1

        model.learn(total_timesteps=TIMESTEPS, reset_num_timesteps=False)
        model.save(f"{modeldir}/model_{sb3_algo}_{TIMESTEPS*iters}")


def test(env, sb3_algo, path_to_model):
    match sb3_algo:
        case 'DDPG':
            model = TD3.load(path_to_model, env=env)
        case 'PPO':
            model = PPO.load(path_to_model, env=env)
        case _:
            print('No alg')
            return
    
    obs = env.reset()[0]
    done = False
    extra_steps = 500

    while True: 
        action, _ = model.predict(obs)
        obs, _, done, _, _ = env.step(action)

        if done:
            extra_steps -= 1

            if extra_steps < 0:
                break

if __name__ == '__main__':
    
    parser = argparse.ArgumentParser(description='Train or test a model')
    parser.add_argument('gymenv', help='BipedalWalker-v3')
    parser.add_argument('sb3_algo', help='TD3, PPO')
    parser.add_argument('-t', '--train', action='store_true')
    parser.add_argument('-s', '--test', metavar='path_to_model')
    args = parser.parse_args()

    if args.train:
        gymenv = gym.make(args.gymenv, render_mode = None)
        train(gymenv, args.sb3_algo)

    if args.test:
        if os.path.isfile(args.test):
            gymenv = gym.make(args.gymenv, render_mode = 'human')
            test(gymenv, args.sb3_algo, path_to_model=args.test)
        else:
            print(f'{args.test} does not exist')
