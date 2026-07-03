import pybullet
import gymnasium as gym

from stable_baselines3 import PPO
from stable_baselines3.common.evaluation import evaluate_policy

# Create the environment
env = gym.make("", render_mode="human") # NOTE: THIS IS SUBJECT TO CHANGE SINCE A CUSTOM ENVIRONMENT WILL BE MADE.
model = PPO("MultiInputPolicy", env, verbose=1) # NOTE: CHANGED FROM MLP POLICY TO MULTI INPUT POLICY SINCE A DICT IS USED FOR OBSERVATIONS
TARGET_REWARD = 10000 # NOTE: THIS IS SUBJECT TO CHANGE BASED ON HOW I HAVE MADE THE ENVIRONMENT FOR THE PROJECT

for i in range(10000):

    model.learn(total_timesteps=10000)
    model.save("WalkerPPO")
    mean_reward, std_reward = evaluate_policy(model, env, n_eval_episodes=10)
    print(f"Mean reward: {mean_reward}")

    if mean_reward >= TARGET_REWARD:
        print("Target reward reached!")
        break

del model
