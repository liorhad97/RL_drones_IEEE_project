#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Example script for training and testing the RL drone system.

This script demonstrates how to use the system with minimal configuration.
It loads the configuration from config.yaml and runs a training session
followed by a testing session.
"""

import os
import yaml
import argparse
import numpy as np
from stable_baselines3 import SAC
from stable_baselines3.common.env_util import make_vec_env

from gym_pybullet_drones.envs.HoverAviary import HoverAviary
from gym_pybullet_drones.utils.enums import ObservationType, ActionType

# Import custom modules
from env_wrapper import EnhancedDroneEnv
from callbacks import HumanFeedbackCallback, GoalAdjustmentCallback, LoggingCallback, CombinedCallback
from models import ModelFactory
from utils import setup_logging, ensure_dir, load_training_config

def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description='RL Drone System Example')
    
    parser.add_argument('--config', type=str, default='config.yaml',
                        help='Path to configuration file')
    parser.add_argument('--train', action='store_true',
                        help='Run training')
    parser.add_argument('--test', action='store_true',
                        help='Run testing')
    parser.add_argument('--timesteps', type=int, default=None,
                        help='Override training timesteps')
    parser.add_argument('--model_path', type=str, default=None,
                        help='Path to model for testing or continuing training')
    
    return parser.parse_args()

def main():
    """Main function."""
    # Parse arguments
    args = parse_args()
    
    # Load configuration
    config = load_training_config(args.config)
    
    # Setup logging
    log_level = config['logging']['log_level'].upper()
    logger = setup_logging(
        log_level=getattr(logging, log_level),
        log_file=os.path.join(config['logging']['log_dir'], 'rl_drone.log')
    )
    logger.info("Starting RL Drone System Example")
    
    # Create base environment
    base_env = make_vec_env(
        HoverAviary,
        env_kwargs=dict(
            obs=ObservationType.KIN, 
            act=ActionType.RPM,
            gui=config['environment']['gui'],
        ),
        n_envs=1
    )
    
    # Create enhanced environment
    env = EnhancedDroneEnv(
        base_env,
        noise_std=config['environment']['noise_std'],
        use_lidar=config['environment']['use_lidar'],
        goal_difficulty=config['environment']['goal_difficulty']
    )
    
    # Create model factory
    model_factory = ModelFactory()
    
    # Override config with command line args if provided
    if args.timesteps is not None:
        config['training']['timesteps'] = args.timesteps
    
    if args.model_path is not None:
        model_path = args.model_path
    else:
        model_path = os.path.join(config['output']['model_dir'], f"{config['training']['algorithm']}_model")
    
    # Training
    if args.train:
        # Create callbacks
        callbacks = []
        
        # Add human feedback callback
        if config['human_feedback']['enabled']:
            human_feedback_callback = HumanFeedbackCallback(
                feedback_frequency=config['human_feedback']['feedback_frequency'],
                feedback_keys=config['human_feedback']['feedback_keys'],
                feedback_ui_enabled=config['human_feedback']['feedback_ui_enabled'],
                verbose=1
            )
            callbacks.append(human_feedback_callback)
        
        # Add goal adjustment callback
        if config['goal_adjustment']['enabled']:
            goal_adjustment_callback = GoalAdjustmentCallback(
                evaluation_frequency=config['goal_adjustment']['evaluation_frequency'],
                success_threshold=config['goal_adjustment']['success_threshold'],
                difficulty_levels=config['goal_adjustment']['difficulty_levels'],
                start_difficulty=config['goal_adjustment']['start_difficulty'],
                verbose=1
            )
            callbacks.append(goal_adjustment_callback)
        
        # Add logging callback
        logging_callback = LoggingCallback(
            log_frequency=config['logging']['log_frequency'],
            log_dir=config['logging']['log_dir'],
            save_frequency=config['logging']['save_frequency'],
            verbose=1
        )
        callbacks.append(logging_callback)
        
        # Combine callbacks
        combined_callback = CombinedCallback(callbacks)
        
        # Create or load model based on algorithm
        algorithm = config['training']['algorithm'].lower()
        
        if os.path.exists(model_path) and args.model_path is not None:
            # Load existing model
            if algorithm == 'sac':
                model = SAC.load(model_path, env=env)
                logger.info(f"Loaded existing SAC model from {model_path}")
            else:
                model = model_factory.load_model(model_path, algorithm, env)
                logger.info(f"Loaded existing {algorithm.upper()} model from {model_path}")
        else:
            # Create new model
            if algorithm == 'sac':
                model = model_factory.create_sac_model(
                    env=env,
                    learning_rate=config['training']['learning_rate'],
                    buffer_size=config['training']['buffer_size'],
                    batch_size=config['training']['batch_size'],
                    tau=config['training']['tau'],
                    gamma=config['training']['gamma'],
                    train_freq=config['training']['train_freq'],
                    gradient_steps=config['training']['gradient_steps'],
                    learning_starts=config['training']['learning_starts'],
                    policy_kwargs=config['model']['policy_kwargs'],
                    verbose=1,
                    seed=config['training']['seed']
                )
            elif algorithm == 'td3':
                model = model_factory.create_td3_model(
                    env=env,
                    learning_rate=config['training']['learning_rate'],
                    buffer_size=config['training']['buffer_size'],
                    batch_size=config['training']['batch_size'],
                    tau=config['training']['tau'],
                    gamma=config['training']['gamma'],
                    policy_delay=config['training']['policy_delay'],
                    noise_type=config['training']['noise_type'],
                    noise_std=config['training']['noise_std'],
                    policy_kwargs=config['model']['policy_kwargs'],
                    verbose=1,
                    seed=config['training']['seed']
                )
            elif algorithm == 'ppo':
                model = model_factory.create_ppo_model(
                    env=env,
                    learning_rate=config['training']['learning_rate'],
                    n_steps=config['training']['n_steps'],
                    batch_size=config['training']['batch_size'],
                    n_epochs=config['training']['n_epochs'],
                    gamma=config['training']['gamma'],
                    clip_range=config['training']['clip_range'],
                    policy_kwargs=config['model']['policy_kwargs'],
                    verbose=1,
                    seed=config['training']['seed']
                )
            elif algorithm == 'a2c':
                model = model_factory.create_a2c_model(
                    env=env,
                    learning_rate=config['training']['learning_rate'],
                    n_steps=config['training']['n_steps'],
                    gamma=config['training']['gamma'],
                    gae_lambda=config['training']['gae_lambda'],
                    ent_coef=config['training']['ent_coef'],
                    vf_coef=config['training']['vf_coef'],
                    policy_kwargs=config['model']['policy_kwargs'],
                    verbose=1,
                    seed=config['training']['seed']
                )
            else:
                raise ValueError(f"Unsupported algorithm: {algorithm}")
            
            logger.info(f"Created new {algorithm.upper()} model")
        
        # Ensure directory exists
        ensure_dir(os.path.dirname(model_path))
        
        # Train the model
        logger.info(f"Training for {config['training']['timesteps']} timesteps")
        model.learn(
            total_timesteps=config['training']['timesteps'],
            callback=combined_callback
        )
        
        # Save the model
        model.save(model_path)
        logger.info(f"Model saved to {model_path}")
    
    # Testing
    if args.test:
        # Load model if not already loaded during training
        if not args.train or not locals().get('model'):
            algorithm = config['training']['algorithm'].lower()
            
            try:
                if algorithm == 'sac':
                    model = SAC.load(model_path, env=env)
                else:
                    model = model_factory.load_model(model_path, algorithm, env)
                
                logger.info(f"Loaded {algorithm.upper()} model from {model_path}")
            except Exception as e:
                logger.error(f"Failed to load model: {e}")
                return
        
        # Run test episodes
        logger.info("Starting test episodes")
        episodes = 5
        
        for episode in range(episodes):
            logger.info(f"Episode {episode+1}/{episodes}")
            obs = env.reset()
            done = False
            total_reward = 0
            step = 0
            
            while not done and step < 1000:
                # Get action from model
                action, _states = model.predict(obs, deterministic=True)
                
                # Take action in environment
                obs, reward, done, info = env.step(action)
                total_reward += reward
                
                # Check if goal is reached
                if hasattr(env, 'goal'):
                    position = obs[:3]
                    distance_to_goal = np.linalg.norm(position - env.goal)
                    
                    if distance_to_goal < 0.1:
                        logger.info(f"Goal reached at step {step}!")
                        # You could inject a new goal here
                        # env.set_goal('medium')
                
                step += 1
            
            logger.info(f"Episode {episode+1} finished after {step} steps with reward {total_reward:.2f}")
        
        logger.info("Testing completed")

if __name__ == "__main__":
    import logging
    main()
