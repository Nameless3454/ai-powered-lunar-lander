# AI-Powered Lunar Lander

Train an AI agent to control and land a simulated lunar spacecraft using **Reinforcement Learning** and **Proximal Policy Optimization (PPO)**.

## Overview

This project explores how an AI agent can learn to control a lunar lander through trial and error. Using the Gymnasium LunarLander-v3 environment and Stable-Baselines3, the agent learns a landing strategy by interacting with the simulation and receiving rewards or penalties.

## Features

- Reinforcement learning with PPO
- Lunar landing simulation using Gymnasium
- AI model training and saving
- Episode reward evaluation
- Simulation visualization
- TensorBoard logging for monitoring training progress

## Technologies Used

- Python
- Gymnasium
- Stable-Baselines3
- PyTorch
- NumPy
- TensorBoard
- Box2D

## Project Structure

```text
ai-powered-lunar-lander/
├── models/                 # Saved trained models
├── logs/                   # Training and evaluation logs
├── simulate_scene.py       # Visual simulation
├── evaluate.py             # Model evaluation
├── requirements.txt        # Python dependencies
├── .gitignore
└── README.md
```

*Note: Update the project structure above to match your actual files.*

## Getting Started

### 1. Clone the repository

```bash
git clone https://github.com/Nameless3454/ai-powered-lunar-lander.git
cd ai-powered-lunar-lander
```

### 2. Create a virtual environment

```bash
python3 -m venv .venv
source .venv/bin/activate
```

### 3. Install dependencies

```bash
python -m pip install --upgrade pip
pip install -r requirements.txt
```

**Note:** The Box2D dependency may require additional system packages, including SWIG, depending on your Python version and operating system.

### 4. Train the AI agent

Run your training script:

```bash
python train.py
```

Make sure `train.py` matches the actual name of your training file.

### 5. Evaluate the trained model

```bash
python evaluate.py --episodes 100
```

This command assumes your `evaluate.py` script supports the `--episodes` argument.

### 6. Run the visual simulation

```bash
python simulate_scene.py
```

Check that the script loads the trained model before running the simulation.

## How It Works

1. **Environment:** The lander operates inside the LunarLander-v3 simulation.
2. **Observation:** The agent receives information about the lander's state.
3. **Action:** The agent selects an action to control the lander.
4. **Reward:** The environment provides rewards or penalties based on the outcome.
5. **Learning:** PPO updates the agent's policy using its experience.
6. **Evaluation:** The trained agent is tested across multiple episodes.

## Model and Performance

The project uses PPO, a reinforcement learning algorithm, to learn a landing policy.

Training performance depends on factors such as the number of training steps, hyperparameters, and random seeds.

*Add your actual evaluation results here after training, such as mean reward over 100 episodes and the number of successful landings.*

## Future Improvements

- Compare different learning rates and discount factors.
- Track reward improvements during training.
- Evaluate landing success rates over many episodes.
- Compare PPO with other reinforcement learning algorithms.
- Improve training efficiency and model performance.

## Learning Goals

- Understand reinforcement learning fundamentals.
- Implement and evaluate a PPO agent.
- Work with simulated environments using Gymnasium.
- Monitor and improve AI training experiments.

## Author

**Kabeer Prajapati**

GitHub: [@Nameless3454](https://github.com/Nameless3454)

---

If you find this project useful, consider giving the repository a star!
