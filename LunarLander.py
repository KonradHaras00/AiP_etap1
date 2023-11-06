import gym
from stable_baselines3 import A2C

models_dir = "models/A2C"

env = gym.make('LunarLander-v2', render_mode='human')  # continuous: LunarLanderContinuous-v2
env.reset()

model_path = f"{models_dir}/90000.zip"
model = A2C.load(model_path, env=env)


episodes = 5

for ep in range(episodes):
	obs, info = env.reset()
	done = False
	print(ep)
	while not done:
		action, _states = model.predict(obs)
		obs, rewards, done, trunced, info = env.step(action)
		env.render()

env.close()


