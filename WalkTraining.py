import pybullet
import gymnasium as gym
from walkEnv import *
from standEnv import *
from stable_baselines3 import PPO
from stable_baselines3.common.evaluation import evaluate_policy
from stable_baselines3.common.vec_env import SubprocVecEnv

RENDER_MODE = "direct" # human / direct

def main():
    def make_env():
        # Register the environment so we can create it with gym.make()
        gym.register(
            id="gymnasium_env/Walker-v0",
            entry_point=WalkerEnv,
        )
        return gym.make("gymnasium_env/Walker-v0")

    # Create the environment
    env = SubprocVecEnv([make_env for _ in range(4)])
    model = PPO.load("StandPPO", env=env)

    model.learn(total_timesteps=2500000)
    model.save("WalkerPPO")
    mean_reward, std_reward = evaluate_policy(model, env, n_eval_episodes=10)
    print(f"Mean reward: {mean_reward}")
    print(f"Std reward: {std_reward}")

    env.close()

if __name__ == "__main__":
    main()
