
import gymnasium as gym
from stable_baselines3 import PPO
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.evaluation import evaluate_policy

# Create folders automatically when saving the model
import os
os.makedirs("models", exist_ok=True)
os.makedirs("logs", exist_ok=True)

# Create and monitor the Lunar Lander environment
env = gym.make("LunarLander-v3", render_mode="human")
env = Monitor(env, "logs/baseline")

# Create the PPO reinforcement learning agent
model = PPO(
    "MlpPolicy",
    env,
    learning_rate=3e-4,
    n_steps=1024,
    batch_size=64,
    gamma=0.99,
    verbose=1,
    tensorboard_log="./logs/tensorboard/"
)

# Train the agent
model.learn(
    total_timesteps=100_000,
    tb_log_name="PPO_baseline"
)

# Save the trained model
model.save("models/ppo_baseline")

# Evaluate the agent on 10 episodes
eval_env = gym.make("LunarLander-v3")
mean_reward, std_reward = evaluate_policy(
    model,
    eval_env,
    n_eval_episodes=10,
    deterministic=True
)

print(f"Mean reward over 10 episodes: {mean_reward:.2f}")
print(f"Reward standard deviation: {std_reward:.2f}")

env.close()
eval_env.close()

print("Training complete! Model saved in models/ppo_baseline.zip")
