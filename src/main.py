#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Main entry point for the RL Drone System.

This script handles command-line arguments, initializes the environment, 
trains or tests the model, and provides a user interface for human feedback.

Usage:
    python main.py --train --timesteps 100000 --goal_difficulty medium
    python main.py --test --model_path models/hover_drone_SAC_model
"""

import os
import argparse
import numpy as np
from stable_baselines3 import SAC
from stable_baselines3.common.env_util import make_vec_env

from gym_pybullet_drones.envs.HoverAviary import HoverAviary
from gym_pybullet_drones.utils.enums import ObservationType, ActionType

# Import custom modules
from env_wrapper import EnhancedDroneEnv
from callbacks import HumanFeedbackCallback
from utils import setup_logging, ensure_dir

def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description='RL Drone System')
    
    # Mode arguments
    mode_group = parser.add_mutually_exclusive_group(required=True)
    mode_group.add_argument('--train', action='store_true', help='Train mode')
    mode_group.add_argument('--test', action='store_true', help='Test mode')
    
    # Training arguments
    parser.add_argument('--timesteps', type=int, default=100000, 
                        help='Total timesteps for training')
    parser.add_argument('--goal_difficulty', type=str, default='medium',
                        choices=['easy', 'medium', 'hard'],
                        help='Difficulty level for goal')
    parser.add_argument('--noise_std', type=float, default=0.01,
                        help='Standard deviation of sensor noise')
    
    # Model arguments
    parser.add_argument('--model_path', type=str, default='models/hover_drone_SAC_model',
                        help='Path to save/load model')
    parser.add_argument('--learning_rate', type=float, default=0.0003,
                        help='Learning rate for the model')
    parser.add_argument('--buffer_size', type=int, default=1000000,
                        help='Size of the replay buffer')
    
    # Environment arguments
    parser.add_argument('--use_lidar', action='store_true', default=True,
                        help='Use lidar mapping for state abstraction')
    parser.add_argument('--gui', action='store_true', default=True,
                        help='Enable GUI for the environment')
    
    return parser.parse_args()

def create_environment(args):
    """Create and configure the environment."""
    # Create the base environment
    base_env = make_vec_env(
        HoverAviary,
        env_kwargs=dict(
            obs=ObservationType.KIN, 
            act=ActionType.RPM,
            gui=args.gui,
        ),
        n_envs=1
    )
    
    # Wrap the environment with our enhanced functionality
    enhanced_env = EnhancedDroneEnv(
        base_env, 
        noise_std=args.noise_std, 
        use_lidar=args.use_lidar, 
        goal_difficulty=args.goal_difficulty
    )
    
    return enhanced_env

def configure_model(args, env):
    """Configure the RL model."""
    # Custom model parameters
    model_params = {
        'policy': 'MlpPolicy',
        'learning_rate': args.learning_rate,
        'buffer_size': args.buffer_size,
        'learning_starts': 100,
        'batch_size': 256,
        'tau': 0.005,  # Target update rate
        'gamma': 0.99,  # Discount factor
        'train_freq': 1,
        'gradient_steps': 1,
        'action_noise': None,  # We're adding noise in our environment
        'verbose': 1
    }
    
    # Initialize or load the model
    try:
        model = SAC.load(args.model_path, env=env)
        print(f"Loaded existing model from {args.model_path}")
    except:
        print(f"No existing model found at {args.model_path}, creating new model")
        model = SAC(**model_params, env=env)
    
    return model

def train_model(args, env, model):
    """Train the model."""
    # Human feedback callback
    human_feedback_callback = HumanFeedbackCallback(verbose=1)
    
    # Ensure directory exists
    ensure_dir(os.path.dirname(args.model_path))
    
    # Train the agent with human feedback
    model.learn(total_timesteps=args.timesteps, callback=human_feedback_callback)
    
    # Save the trained model
    model.save(args.model_path)
    print(f"Model saved to {args.model_path}")

def test_model(args, env, model):
    """Test the trained model."""
    # Reset environment
    obs = env.reset()
    
    # Run test episodes
    episodes = 5
    for episode in range(episodes):
        print(f"\nEpisode {episode+1}/{episodes}")
        done = False
        total_reward = 0
        step = 0
        
        # Reset for new episode
        obs = env.reset()
        
        while not done and step < 1000:
            # Get action from model
            action, _states = model.predict(obs, deterministic=True)
            
            # Take action in environment
            obs, reward, done, info = env.step(action)
            total_reward += reward
            
            # Print status
            if step % 50 == 0:
                print(f"Step {step}: Reward = {reward:.3f}, Total = {total_reward:.3f}")
                
                # Check if goal is finished
                if hasattr(env, 'goal'):
                    # Extract position from observation (assuming first 3 elements are x,y,z)
                    position = obs[:3]
                    distance_to_goal = np.linalg.norm(position - env.goal)
                    print(f"Distance to goal: {distance_to_goal:.3f}")
                    
                    if distance_to_goal < 0.1:
                        print(f"Goal achieved at step {step}!")
            
            step += 1
            
            if done:
                print(f"Episode finished after {step} steps with total reward: {total_reward:.3f}")
                break
                
        print(f"Episode {episode+1} complete: Total steps = {step}, Total reward = {total_reward:.3f}")

def main():
    """Main function."""
    # Parse arguments
    args = parse_args()
    
    # Setup logging
    logger = setup_logging()
    logger.info("Starting RL Drone System")
    logger.info(f"Arguments: {args}")
    
    # Create environment
    env = create_environment(args)
    logger.info("Environment created")
    
    # Configure model
    model = configure_model(args, env)
    logger.info("Model configured")
    
    # Train or test based on mode
    if args.train:
        logger.info("Starting training")
        train_model(args, env, model)
    else:  # Test mode
        logger.info("Starting testing")
        test_model(args, env, model)
    
    logger.info("RL Drone System completed")

if __name__ == "__main__":
    main()
