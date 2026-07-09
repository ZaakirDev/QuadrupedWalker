import pybullet
import gymnasium as gym
from walkEnv import *
from standEnv import *
from stable_baselines3 import PPO
from stable_baselines3.common.evaluation import evaluate_policy

# Selection
select = int(input("1. Walker\n2. Stand"))
if select == 1:
    Mode = "Walker"
    gym.register(
        id=f"gymnasium_env/Walker-v0",
        entry_point=WalkerEnv,
    )
else:
    Mode = "Stand"
    gym.register(
        id=f"gymnasium_env/Stand-v0",
        entry_point=StandEnv,
    )

# Create the environment
env = gym.make(f"gymnasium_env/{Mode}-v0")
model = PPO("MultiInputPolicy", env, verbose=1)
# model = PPO.load(f"{Mode}PPO", env=env)

model.learn(total_timesteps=100000)
# model.save("WalkerPPO")
mean_reward, std_reward = evaluate_policy(model, env, n_eval_episodes=10)
print(f"Mean reward: {mean_reward}")
print(f"Std reward: {std_reward}")

env.close()