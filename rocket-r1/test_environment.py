
import gymnasium as gym

env = gym.make("LunarLander-v3", render_mode="human")

observation, info = env.reset(seed=42)

print("Initial observation:", observation)
print("Number of observations:", len(observation))
print("Available actions:", env.action_space.n)

for step in range(1000):
    action = env.action_space.sample()

    observation, reward, terminated, truncated, info = (
        env.step(action)
    )

    if terminated or truncated:
        print("Episode ended after", step + 1, "steps")
        break

env.close()
