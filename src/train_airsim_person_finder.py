#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
AirSim Training Script for Person Finder Drone

This script implements training and testing of the reinforcement learning
agent for person finding missions with drones in AirSim.
"""

import os
import argparse
import numpy as np
import cv2
import gymnasium as gym
from stable_baselines3 import SAC
from stable_baselines3.common.env_util import make_vec_env

# Import AirSim environment wrappers
from airsim_wrapper import AirSimDroneEnv
from airsim_person_finder import AirSimPersonFinderEnv

# Import person detection utilities
from person_detection_utils import create_target_person_image

# Import utilities for logging and callbacks
from utils import setup_logging, ensure_dir
from callbacks import HumanFeedbackCallback, LoggingCallback, CombinedCallback

def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description='AirSim Person Finder Drone Training')
    
    # Mode arguments
    parser.add_argument('--train', action='store_true', help='Train the agent')
    parser.add_argument('--test', action='store_true', help='Test the agent')
    
    # Environment arguments
    parser.add_argument('--ip_address', type=str, default="127.0.0.1", 
                        help='IP address of the AirSim server')
    parser.add_argument('--goal_type', type=str, choices=['image', 'text'], default='image',
                      help='Type of goal description (image or text)')
    parser.add_argument('--target_image', type=str, default=None,
                      help='Path to target person image')
    parser.add_argument('--target_description', type=str, default=None,
                      help='Description of target person')
    parser.add_argument('--simulate_detection', action='store_true', default=True,
                       help='Simulate person detection instead of using models')
    
    # Training arguments
    parser.add_argument('--timesteps', type=int, default=100000,
                      help='Total timesteps for training')
    parser.add_argument('--model_path', type=str, default='models/airsim_person_finder_model',
                      help='Path to save/load model')
    parser.add_argument('--human_feedback', action='store_true', default=True,
                      help='Enable human feedback during training')
    
    return parser.parse_args()

def create_environment(args):
    """Create and configure the AirSim person finder environment."""
    # Create base AirSim environment
    base_env = AirSimDroneEnv(
        ip_address=args.ip_address,
        drone_name="PX4",
        camera_name="front_center", 
        image_shape=(480, 640, 3)  # Height, width, channels
    )
    
    # Wrap with person finder environment
    env = AirSimPersonFinderEnv(
        base_env,
        use_visual_goal=(args.goal_type == 'image'),
        detection_threshold=0.5,
        match_threshold=0.7,
        goal_difficulty='medium',
        camera_resolution=(640, 480),  # Width, height
        simulate_detection=args.simulate_detection
    )
    
    # Load target person image if provided
    target_image = None
    if args.target_image and os.path.exists(args.target_image):
        target_image = cv2.imread(args.target_image)
        if target_image is None:
            print(f"Warning: Could not load target image from {args.target_image}")
    
    # Set the person goal
    if args.goal_type == 'image':
        if target_image is None:
            # Create a simple target image if none provided
            print("Creating default target person image")
            target_image = create_target_person_image(color=(0, 0, 255))  # Red person
        env.set_person_goal(target_image=target_image)
    else:  # Text description
        if not args.target_description:
            args.target_description = "Person wearing a red shirt and blue jeans"
            print(f"Using default target description: {args.target_description}")
        env.set_person_goal(target_description=args.target_description)
    
    return env

def setup_callbacks(args):
    """Set up training callbacks."""
    callbacks = []
    
    # Human feedback callback
    if args.human_feedback:
        human_feedback_callback = HumanFeedbackCallback(
            feedback_frequency=20,
            feedback_keys={
                '+': 1.0,   # Positive feedback (good identification)
                '-': -1.0,  # Negative feedback (wrong identification)
                '0': 0.0    # Neutral feedback
            },
            feedback_ui_enabled=False,  # No separate UI window
            verbose=1
        )
        callbacks.append(human_feedback_callback)
    
    # Logging callback
    log_dir = os.path.join('logs', 'airsim_person_finder')
    ensure_dir(log_dir)
    logging_callback = LoggingCallback(
        log_frequency=1000,
        log_dir=log_dir,
        save_frequency=10000,
        verbose=1
    )
    callbacks.append(logging_callback)
    
    # Combine all callbacks
    return CombinedCallback(callbacks)

def train(env, args):
    """Train the person finder agent in AirSim."""
    # Initialize or load model
    try:
        model = SAC.load(args.model_path, env=env)
        print(f"Loaded existing model from {args.model_path}")
    except:
        print(f"Creating new SAC model")
        model = SAC(
            policy='MlpPolicy',
            env=env,
            verbose=1,
            buffer_size=1000000,
            learning_rate=0.0003,
            batch_size=256,
            tau=0.005,
            gamma=0.99,
            train_freq=1,
            gradient_steps=1,
        )
    
    # Set up callbacks
    callbacks = setup_callbacks(args)
    
    # Train the agent
    print(f"Training for {args.timesteps} timesteps...")
    model.learn(total_timesteps=args.timesteps, callback=callbacks)
    
    # Save the model
    ensure_dir(os.path.dirname(args.model_path))
    model.save(args.model_path)
    print(f"Model saved to {args.model_path}")
    
    return model

def test(env, args):
    """Test the trained person finder agent in AirSim."""
    # Load model
    try:
        model = SAC.load(args.model_path, env=env)
        print(f"Loaded model from {args.model_path}")
    except:
        print(f"Error: Could not load model from {args.model_path}")
        return
    
    # Run test episodes
    episodes = 5
    max_steps = 1000
    
    for episode in range(episodes):
        print(f"\nEpisode {episode+1}/{episodes}")
        obs, info = env.reset()
        
        done = False
        episode_reward = 0
        step = 0
        
        while not done and step < max_steps:
            # Get action from policy
            action, _states = model.predict(obs, deterministic=True)
            
            # Take action
            next_obs, reward, done, info = env.step(action)
                
            episode_reward += reward
            obs = next_obs
            
            # Print status every 50 steps
            if step % 50 == 0 or info.get('person_found', False):
                print(f"Step {step}: Reward={reward:.2f}, Total={episode_reward:.2f}")
                print(f"Match score: {info.get('match_score', 0):.2f}, "
                     f"Detections: {info.get('num_detections', 0)}")
                
                if info.get('person_found', False):
                    print("TARGET PERSON FOUND!")
                    
                # Create visualization
                vis_img = env.visualize_detection()
                if vis_img is not None:
                    cv2.imshow("Person Detection", vis_img)
                    cv2.waitKey(1)
            
            step += 1
        
        print(f"Episode {episode+1} finished: Steps={step}, Total reward={episode_reward:.2f}")
        
        # Close visualization window
        cv2.destroyAllWindows()

def main():
    """Main function."""
    # Parse arguments
    args = parse_args()
    
    # Setup logging
    logger = setup_logging(log_file='logs/airsim_person_finder.log')
    logger.info("Starting AirSim Person Finder Drone")
    
    # Create output directories
    ensure_dir('models')
    ensure_dir('logs')
    
    # Create environment
    env = create_environment(args)
    
    # Train or test based on mode
    if args.train:
        model = train(env, args)
        
        # Optionally test after training
        if args.test:
            test(env, args)
    
    elif args.test:
        test(env, args)
    
    else:
        print("Please specify either --train or --test")

if __name__ == "__main__":
    main()
