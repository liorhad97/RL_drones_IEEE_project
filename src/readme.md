# Reinforcement Learning Drone System

This repository implements a reinforcement learning architecture for drone control with human feedback, noise filtering, and lidar-based state abstraction as shown in the system diagram.

## Architecture Overview

![Architecture Diagram](architecture.png)

The system consists of the following components:

1. **RLHF (Reinforcement Learning from Human Feedback)**: Incorporates human evaluation into the training process
2. **Goal Definition**: Dynamic goal injection system to adjust simulation parameters
3. **Agent Drone**: The RL model (using Soft Actor-Critic) that controls the drone
4. **Replay Buffer**: Stores experiences for training with extensions for custom data
5. **Noise Filter**: Reduces sensor noise to improve state representation
6. **Lidar Mapping**: Provides state abstraction based on environment scanning
7. **Environment**: The simulation environment for the drone (using PyBullet)

## Project Structure

```
├── main.py                # Entry point for training and testing
├── env_wrapper.py         # Enhanced environment wrapper with additional features
├── replay_buffer.py       # Custom replay buffer implementation
├── callbacks.py           # Human feedback callback and other callbacks
├── models.py              # Model configuration and training utilities
├── utils.py               # Utility functions
└── README.md              # Project documentation
```

## Installation

```bash
# Clone the repository
git clone https://github.com/yourusername/rl-drone-system.git
cd rl-drone-system

# Install dependencies
pip install gymnasium numpy stable-baselines3 gym-pybullet-drones
```

## Usage

### Training the model

```bash
python main.py --train --timesteps 100000 --goal_difficulty medium
```

### Testing the model

```bash
python main.py --test --model_path models/hover_drone_SAC_model
```

### Providing Human Feedback

During training, you can provide human feedback through the keyboard:

- Press `+` to provide positive feedback
- Press `-` to provide negative feedback
- Press `0` to provide neutral feedback

## Components Details

### Enhanced Environment Wrapper

The environment wrapper adds several features on top of the base environment:
- Custom goal definition with different difficulty levels
- Noise simulation and filtering for realistic sensor data
- Lidar-based state abstraction
- Integration of human feedback into the reward function

### Human Feedback System

The human feedback system allows for:
- Real-time evaluation of drone performance
- Integration of human preferences into the reward function
- Correction of undesirable behaviors

### Replay Buffer

The enhanced replay buffer provides:
- Storage of experiences for training
- Custom data structures for additional information
- Integration with the SAC algorithm

### Goal Definition System

The goal system allows for:
- Dynamic adjustment of target positions
- Different difficulty levels
- Evaluation of goal completion

## Extending the System

### Adding New Goals

To add new goal types, modify the `set_goal` method in `env_wrapper.py`:

```python
def set_goal(self, difficulty, custom_goal=None):
    if custom_goal is not None:
        self.goal = custom_goal
    elif difficulty == 'your_new_difficulty':
        self.goal = np.array([x, y, z])  # Your custom goal
```

### Customizing Feedback

To customize the feedback mechanism, modify the `HumanFeedbackCallback` class in `callbacks.py`.

## License

MIT License

## Contributors

- Your Name
