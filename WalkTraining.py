import pybullet
import gymnasium as gym
from walkEnv import *
from stable_baselines3 import PPO
from stable_baselines3.common.evaluation import evaluate_policy
from stable_baselines3.common.vec_env import SubprocVecEnv

def make_env():
    # Register the environment so we can create it with gym.make()
    gym.register(
        id="gymnasium_env/Walker-v0",
        entry_point=WalkerEnv,
    )
    return gym.make("gymnasium_env/Walker-v0")

def main():
    # Create the environment
    env = SubprocVecEnv([make_env for _ in range(4)])
    model = PPO("MultiInputPolicy", env, verbose=1)

    model.learn(total_timesteps=1250000)
    model.save("./Models/WalkerPPO")
    mean_reward, std_reward = evaluate_policy(model, env, n_eval_episodes=10)
    print(f"Mean reward: {mean_reward}")
    print(f"Std reward: {std_reward}")

    env.close()

if __name__ == "__main__":
    main()
