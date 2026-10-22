
import argparse
import csv
import os
import time

import gymnasium as gym
import numpy as np
from stable_baselines3 import PPO


DEFAULT_MODEL = "models/ppo_baseline.zip"


def main():
    parser = argparse.ArgumentParser(
        description="Evaluate a trained PPO Lunar Lander agent without a UI."
    )
    parser.add_argument("--episodes", type=int, default=100)
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument("--csv", default="evaluation_results.csv")
    parser.add_argument("--seed", type=int, default=1000)
    args = parser.parse_args()

    if args.episodes < 1:
        parser.error("--episodes must be at least 1")

    if not os.path.isfile(args.model):
        parser.error(f"Model not found: {args.model}")

    print("\n" + "=" * 60)
    print("       LUNAR LANDER | HEADLESS EVALUATION")
    print("=" * 60)
    print(f"Model:    {args.model}")
    print(f"Episodes: {args.episodes}")
    print("Graphics: Disabled")
    print("Policy:   Deterministic")
    print("-" * 60)

    # No render_mode means no simulation window is opened.
    env = gym.make("LunarLander-v3")
    model = PPO.load(args.model, device="cpu")

    results = []
    start_time = time.time()

    try:
        for episode in range(1, args.episodes + 1):
            obs, info = env.reset(seed=args.seed + episode)

            total_reward = 0.0
            steps = 0
            terminated = False
            truncated = False

            while not (terminated or truncated):
                action, _ = model.predict(
                    obs, deterministic=True
                )

                obs, reward, terminated, truncated, info = (
                    env.step(action)
                )

                total_reward += float(reward)
                steps += 1

            # The environment has ended. This measures the standard
            # 200-point benchmark, not guaranteed physical safety.
            benchmark_pass = total_reward >= 200.0

            result = {
                "episode": episode,
                "reward": round(total_reward, 3),
                "steps": steps,
                "terminated": terminated,
                "truncated": truncated,
                "benchmark_200": benchmark_pass,
            }
            results.append(result)

            if episode % 10 == 0 or episode == args.episodes:
                recent = results[-min(10, len(results)):]
                recent_avg = np.mean(
                    [r["reward"] for r in recent]
                )
                print(
                    f"Episode {episode:3d}/{args.episodes} | "
                    f"Reward: {total_reward:8.2f} | "
                    f"Steps: {steps:3d} | "
                    f"200+ benchmark: "
                    f"{'YES' if benchmark_pass else 'NO ':3s} | "
                    f"Last 10 avg: {recent_avg:7.2f}"
                )

    finally:
        env.close()

    rewards = np.array([r["reward"] for r in results])
    passed = sum(r["benchmark_200"] for r in results)
    elapsed = time.time() - start_time

    # Export individual episode results.
    with open(args.csv, "w", newline="") as file:
        writer = csv.DictWriter(
            file, fieldnames=list(results[0].keys())
        )
        writer.writeheader()
        writer.writerows(results)

    print("\n" + "=" * 60)
    print("                 FINAL RESULTS")
    print("=" * 60)
    print(f"Episodes evaluated:       {len(results)}")
    print(f"Mean reward:              {np.mean(rewards):.2f}")
    print(f"Reward standard deviation:{np.std(rewards):.2f}")
    print(f"Best episode:             {np.max(rewards):.2f}")
    print(f"Worst episode:            {np.min(rewards):.2f}")
    print(f"Median reward:            {np.median(rewards):.2f}")
    print(
        f"200+ reward episodes:     {passed}/{len(results)} "
        f"({100 * passed / len(results):.1f}%)"
    )
    print(f"Average steps:             {np.mean([r['steps'] for r in results]):.1f}")
    print(f"Elapsed time:              {elapsed:.1f} seconds")
    print(f"CSV saved to:              {args.csv}")
    print("=" * 60)


if __name__ == "__main__":
    main()
