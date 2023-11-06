import gym
from stable_baselines3 import PPO

models_dir = "models/cartPole_PPO"

env = gym.make('CartPole-v1', render_mode='human')  # continuous: LunarLanderContinuous-v2
env.reset()

episodes = 5

model = PPO('MlpPolicy', env, verbose=1)

TIMESTEPS = 10000
iters = 0
while True:
    iters += 1
    
    model.learn(total_timesteps=TIMESTEPS, reset_num_timesteps=False)
    model.save(f"{models_dir}/{TIMESTEPS*iters}")