import pybullet
import gymnasium as gym
from walkEnv import *
from stable_baselines3 import PPO
from stable_baselines3.common.evaluation import evaluate_policy

gym.register(
    id=f"gymnasium_env/Walker-v0",
    entry_point=WalkerEnv,
)

# Create the environment
env = gym.make(f"gymnasium_env/Walker-v0", render_mode="human")
model = PPO.load(f"WalkerPPO", env=env)

model.learn(total_timesteps=100000)
# model.save("WalkerPPO")
mean_reward, std_reward = evaluate_policy(model, env, n_eval_episodes=10)
print(f"Mean reward: {mean_reward}")
print(f"Std reward: {std_reward}")

env.close()