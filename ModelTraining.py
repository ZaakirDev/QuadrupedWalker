import pybullet
import gymnasium as gym
from walkEnv import *
from standEnv import *
from stable_baselines3 import PPO
from stable_baselines3.common.evaluation import evaluate_policy

RENDER_MODE = "human" # human / direct

"""
STANDING PHASE ===================================================================
"""

# Register stand Environment
gym.register(
    id="gymnasium_env/Stand-v0",
    entry_point=StandEnv,
)

# Create the environment
env = gym.make("gymnasium_env/Stand-v0", render_mode=RENDER_MODE)
model = PPO("MultiInputPolicy", env, verbose=1)

model.learn(total_timesteps=1500000)
model.save("StandPPO")
mean_reward, std_reward = evaluate_policy(model, env, n_eval_episodes=10)
print(f"Mean reward: {mean_reward}")
print(f"Std reward: {std_reward}")

env.close()

"""
WALKING PHASE ===================================================================
"""

# Register the environment so we can create it with gym.make()
gym.register(
    id="gymnasium_env/Walker-v0",
    entry_point=WalkerEnv,
)

# Create the environment
env = gym.make("gymnasium_env/Walker-v0", render_mode=RENDER_MODE)
model = PPO.load("StandPPO", env=env)

model.learn(total_timesteps=2500000)
model.save("WalkerPPO")
mean_reward, std_reward = evaluate_policy(model, env, n_eval_episodes=10)
print(f"Mean reward: {mean_reward}")
print(f"Std reward: {std_reward}")

env.close()
