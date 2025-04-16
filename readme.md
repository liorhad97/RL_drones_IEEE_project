# RL Drone Person Finder

This project implements a reinforcement learning system for training a drone to find specific people based on image recognition or textual descriptions.

## Architecture Overview

The system combines several components:
1. **Drone Environment**: Based on PyBullet simulation with enhanced wrapper for goals and sensors
2. **Person Detection**: Computer vision capabilities for detecting and recognizing people
3. **Reinforcement Learning**: SAC algorithm to learn optimal search policies 
4. **Human Feedback**: Integration of human evaluations into training

## File Structure

```
├── person_finder_env.py        # Person finder environment wrapper
├── person_detection_utils.py   # Person detection and recognition utilities
├── callbacks.py                # Training callbacks including human feedback
├── train_person_finder.py      # Training and testing script
├── env_wrapper.py              # Base environment wrapper (using from your files)
├── utils.py                    # Utility functions (using from your files)
└── README.md                   # This file
```

## Installation

1. Clone the repository and install dependencies:

```bash
# Install dependencies
pip install gymnasium numpy stable-baselines3 opencv-python torch
pip install gym-pybullet-drones

# Optional dependencies
pip install scikit-learn matplotlib keyboard
```

## Usage

### Training the Person Finder

To train a person finder drone using image-based goals:

```bash
python train_person_finder.py --train --goal_type image --simulate_detection
```

With a specific target image:

```bash
python train_person_finder.py --train --goal_type image --target_image path/to/person.jpg
```

To train with text-based description:

```bash
python train_person_finder.py --train --goal_type text --target_description "Person wearing a red shirt and blue jeans"
```

### Testing a Trained Model

To test a trained model:

```bash
python train_person_finder.py --test --model_path models/person_finder_model --goal_type image
```

### Human Feedback

During training, you can provide human feedback through keyboard:
- Press `+` to provide positive feedback
- Press `-` to provide negative feedback
- Press `0` to provide neutral feedback

## Components Detail

### Person Finder Environment

The `PersonFinderEnv` class extends the base drone environment with:
- Person detection and recognition capabilities
- Goal-setting based on images or descriptions
- Reward shaping for the person finding task

### Person Detection

The detection system supports:
- Real person detection (using Faster R-CNN)
- Feature extraction for person matching
- Simulated detection mode for easier testing

### RL Training with Human Feedback

The system integrates:
- SAC algorithm for continuous control
- Human feedback to improve learning
- Callbacks for monitoring and logging

## Extending the System

### Adding New Detection Models

To use different detection models:

1. Modify the `_initialize_detector` method in `PersonDetector` class
2. Update the feature extraction in `_initialize_feature_extractor`

### Custom Reward Shaping

To customize rewards for different mission parameters:

1. Edit the `_compute_person_finding_reward` method in `PersonFinderEnv`
2. Add new reward components for specific behaviors

## Troubleshooting

- If you get import errors, ensure all dependencies are installed
- For CUDA out-of-memory errors, try setting `device='cpu'` in the person detector
- If person detection fails, enable `simulate_detection=True` for testing
