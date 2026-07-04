import pybullet
import gymnasium as gym
from Environment import *
from stable_baselines3 import PPO
from stable_baselines3.common.evaluation import evaluate_policy

# Register the environment so we can create it with gym.make()
gym.register(
    id="gymnasium_env/Walker-v0",
    entry_point=WalkerEnv,
)

# Create the environment
env = gym.make("gymnasium_env/Walker-v0")
model = PPO("MultiInputPolicy", env, verbose=1)

model.learn(total_timesteps=10000)
model.save("WalkerPPO")
mean_reward, std_reward = evaluate_policy(model, env, n_eval_episodes=10)
print(f"Mean reward: {mean_reward}")
print(f"Std reward: {std_reward}")

env.close()
