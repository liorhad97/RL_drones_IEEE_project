#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Model Configuration and Training Utilities

This module provides utilities for configuring and training RL models
for the drone system. It implements the "Agent Drone" component (blue box)
in the architecture diagram.
"""

import os
import logging
import numpy as np
from typing import Dict, List, Optional, Union, Any, Tuple, Type
from stable_baselines3 import SAC, PPO, TD3, A2C
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.noise import NormalActionNoise, OrnsteinUhlenbeckActionNoise
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor
import torch as th
import torch.nn as nn
import gymnasium as gym

class CustomCNN(BaseFeaturesExtractor):
    """
    Custom CNN feature extractor for processing image-like observations.
    
    This could be used if the drone has camera input or if the lidar data
    is represented as an image.
    
    Attributes:
        features_dim (int): Dimension of the extracted features
        cnn (nn.Sequential): CNN model for feature extraction
    """
    
    def __init__(self, observation_space: gym.spaces.Box, features_dim: int = 256):
        """
        Initialize the CNN feature extractor.
        
        Args:
            observation_space (gym.spaces.Box): Observation space
            features_dim (int): Dimension of the extracted features
        """
        super(CustomCNN, self).__init__(observation_space, features_dim)
        
        # Determine input channels based on observation shape
        if len(observation_space.shape) == 3:
            n_input_channels = observation_space.shape[0]
        else:
            # Assume flat observations need to be reshaped
            n_input_channels = 1
        
        self.cnn = nn.Sequential(
            nn.Conv2d(n_input_channels, 32, kernel_size=8, stride=4, padding=0),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=4, stride=2, padding=0),
            nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=0),
            nn.ReLU(),
            nn.Flatten(),
        )
        
        # Compute shape by doing one forward pass
        with th.no_grad():
            sample = th.as_tensor(observation_space.sample()[None]).float()
            if len(sample.shape) == 3:
                sample = sample.unsqueeze(0)
            elif len(sample.shape) == 1:
                sample = sample.reshape(1, 1, int(np.sqrt(len(sample))), int(np.sqrt(len(sample))))
            n_flatten = self.cnn(sample).shape[1]
        
        self.linear = nn.Sequential(
            nn.Linear(n_flatten, features_dim),
            nn.ReLU()
        )
        
        # Initialize logger
        self.logger = logging.getLogger('CustomCNN')
        self.logger.info(f"Initialized with features_dim={features_dim}")
    
    def forward(self, observations: th.Tensor) -> th.Tensor:
        """
        Extract features from observations.
        
        Args:
            observations (th.Tensor): Observations
            
        Returns:
            th.Tensor: Extracted features
        """
        # Reshape if needed
        input_tensor = observations
        if len(observations.shape) == 2:
            # Batched flat observations
            n = int(np.sqrt(observations.shape[1]))
            input_tensor = observations.reshape(-1, 1, n, n)
        elif len(observations.shape) == 1:
            # Single flat observation
            n = int(np.sqrt(len(observations)))
            input_tensor = observations.reshape(1, 1, n, n)
        
        return self.linear(self.cnn(input_tensor))


class ModelFactory:
    """
    Factory class for creating different RL models.
    
    This class provides a convenient way to create and configure
    different RL models for the drone system.
    
    Attributes:
        logger (logging.Logger): Logger for the factory
    """
    
    def __init__(self):
        """Initialize the model factory."""
        self.logger = logging.getLogger('ModelFactory')
    
    def create_sac_model(
        self,
        env,
        learning_rate: float = 0.0003,
        buffer_size: int = 1000000,
        batch_size: int = 256,
        tau: float = 0.005,
        gamma: float = 0.99,
        train_freq: int = 1,
        gradient_steps: int = 1,
        learning_starts: int = 100,
        policy_kwargs: Optional[Dict[str, Any]] = None,
        verbose: int = 1,
        seed: Optional[int] = None
    ) -> SAC:
        """
        Create a Soft Actor-Critic (SAC) model.
        
        SAC is an off-policy algorithm suitable for continuous action spaces.
        It uses entropy regularization for exploration.
        
        Args:
            env: Training environment
            learning_rate (float): Learning rate
            buffer_size (int): Size of the replay buffer
            batch_size (int): Batch size for sampling
            tau (float): Target network update rate
            gamma (float): Discount factor
            train_freq (int): Training frequency
            gradient_steps (int): Number of gradient steps
            learning_starts (int): Number of steps before learning starts
            policy_kwargs (Dict[str, Any], optional): Policy network kwargs
            verbose (int): Verbosity level
            seed (int, optional): Random seed
            
        Returns:
            SAC: Configured SAC model
        """
        # Default policy kwargs if none provided
        if policy_kwargs is None:
            policy_kwargs = {
                "net_arch": {
                    "pi": [256, 256],
                    "qf": [256, 256]
                }
            }
        
        # Create model
        model = SAC(
            policy="MlpPolicy",
            env=env,
            learning_rate=learning_rate,
            buffer_size=buffer_size,
            batch_size=batch_size,
            tau=tau,
            gamma=gamma,
            train_freq=train_freq,
            gradient_steps=gradient_steps,
            learning_starts=learning_starts,
            policy_kwargs=policy_kwargs,
            verbose=verbose,
            seed=seed
        )
        
        self.logger.info(f"Created SAC model with learning_rate={learning_rate}, "
                         f"buffer_size={buffer_size}")
        
        return model
    
    def create_td3_model(
        self,
        env,
        learning_rate: float = 0.0003,
        buffer_size: int = 1000000,
        batch_size: int = 100,
        tau: float = 0.005,
        gamma: float = 0.99,
        train_freq: Tuple[int, str] = (1, 'episode'),
        learning_starts: int = 100,
        policy_delay: int = 2,
        noise_type: str = 'normal',
        noise_std: float = 0.1,
        policy_kwargs: Optional[Dict[str, Any]] = None,
        verbose: int = 1,
        seed: Optional[int] = None
    ) -> TD3:
        """
        Create a Twin Delayed DDPG (TD3) model.
        
        TD3 is an off-policy algorithm suitable for continuous action spaces.
        It addresses function approximation error by using double critics.
        
        Args:
            env: Training environment
            learning_rate (float): Learning rate
            buffer_size (int): Size of the replay buffer
            batch_size (int): Batch size for sampling
            tau (float): Target network update rate
            gamma (float): Discount factor
            train_freq (Tuple[int, str]): Training frequency
            learning_starts (int): Number of steps before learning starts
            policy_delay (int): Delay between critic and policy updates
            noise_type (str): Type of action noise ('normal' or 'ou')
            noise_std (float): Standard deviation of action noise
            policy_kwargs (Dict[str, Any], optional): Policy network kwargs
            verbose (int): Verbosity level
            seed (int, optional): Random seed
            
        Returns:
            TD3: Configured TD3 model
        """
        # Default policy kwargs if none provided
        if policy_kwargs is None:
            policy_kwargs = {
                "net_arch": [400, 300]
            }
        
        # Configure action noise
        n_actions = env.action_space.shape[0]
        if noise_type == 'normal':
            action_noise = NormalActionNoise(
                mean=np.zeros(n_actions),
                sigma=noise_std * np.ones(n_actions)
            )
        elif noise_type == 'ou':
            action_noise = OrnsteinUhlenbeckActionNoise(
                mean=np.zeros(n_actions),
                sigma=noise_std * np.ones(n_actions)
            )
        else:
            action_noise = None
            self.logger.warning(f"Unknown noise type '{noise_type}', using no noise")
        
        # Create model
        model = TD3(
            policy="MlpPolicy",
            env=env,
            learning_rate=learning_rate,
            buffer_size=buffer_size,
            batch_size=batch_size,
            tau=tau,
            gamma=gamma,
            train_freq=train_freq,
            learning_starts=learning_starts,
            policy_delay=policy_delay,
            action_noise=action_noise,
            policy_kwargs=policy_kwargs,
            verbose=verbose,
            seed=seed
        )
        
        self.logger.info(f"Created TD3 model with learning_rate={learning_rate}, "
                         f"noise_type={noise_type}, noise_std={noise_std}")
        
        return model
    
    def create_ppo_model(
        self,
        env,
        learning_rate: float = 0.0003,
        n_steps: int = 2048,
        batch_size: int = 64,
        n_epochs: int = 10,
        gamma: float = 0.99,
        gae_lambda: float = 0.95,
        clip_range: float = 0.2,
        policy_kwargs: Optional[Dict[str, Any]] = None,
        verbose: int = 1,
        seed: Optional[int] = None
    ) -> PPO:
        """
        Create a Proximal Policy Optimization (PPO) model.
        
        PPO is an on-policy algorithm that uses clipped surrogate objective
        for stable learning.
        
        Args:
            env: Training environment
            learning_rate (float): Learning rate
            n_steps (int): Number of steps to run for each environment per update
            batch_size (int): Minibatch size for each gradient update
            n_epochs (int): Number of epochs to optimize surrogate
            gamma (float): Discount factor
            gae_lambda (float): Factor for GAE
            clip_range (float): Clipping parameter for PPO
            policy_kwargs (Dict[str, Any], optional): Policy network kwargs
            verbose (int): Verbosity level
            seed (int, optional): Random seed
            
        Returns:
            PPO: Configured PPO model
        """
        # Default policy kwargs if none provided
        if policy_kwargs is None:
            policy_kwargs = {
                "net_arch": [dict(pi=[64, 64], vf=[64, 64])]
            }
        
        # Create model
        model = PPO(
            policy="MlpPolicy",
            env=env,
            learning_rate=learning_rate,
            n_steps=n_steps,
            batch_size=batch_size,
            n_epochs=n_epochs,
            gamma=gamma,
            gae_lambda=gae_lambda,
            clip_range=clip_range,
            policy_kwargs=policy_kwargs,
            verbose=verbose,
            seed=seed
        )
        
        self.logger.info(f"Created PPO model with learning_rate={learning_rate}, "
                         f"n_steps={n_steps}, n_epochs={n_epochs}")
        
        return model
    
    def create_a2c_model(
        self,
        env,
        learning_rate: float = 0.0007,
        n_steps: int = 5,
        gamma: float = 0.99,
        gae_lambda: float = 1.0,
        ent_coef: float = 0.0,
        vf_coef: float = 0.5,
        policy_kwargs: Optional[Dict[str, Any]] = None,
        verbose: int = 1,
        seed: Optional[int] = None
    ) -> A2C:
        """
        Create an Advantage Actor-Critic (A2C) model.
        
        A2C is an on-policy algorithm that uses a value function
        to estimate the advantage function.
        
        Args:
            env: Training environment
            learning_rate (float): Learning rate
            n_steps (int): Number of steps to run for each environment per update
            gamma (float): Discount factor
            gae_lambda (float): Factor for GAE
            ent_coef (float): Entropy coefficient
            vf_coef (float): Value function coefficient
            policy_kwargs (Dict[str, Any], optional): Policy network kwargs
            verbose (int): Verbosity level
            seed (int, optional): Random seed
            
        Returns:
            A2C: Configured A2C model
        """
        # Default policy kwargs if none provided
        if policy_kwargs is None:
            policy_kwargs = {
                "net_arch": [64, 64]
            }
        
        # Create model
        model = A2C(
            policy="MlpPolicy",
            env=env,
            learning_rate=learning_rate,
            n_steps=n_steps,
            gamma=gamma,
            gae_lambda=gae_lambda,
            ent_coef=ent_coef,
            vf_coef=vf_coef,
            policy_kwargs=policy_kwargs,
            verbose=verbose,
            seed=seed
        )
        
        self.logger.info(f"Created A2C model with learning_rate={learning_rate}, "
                         f"n_steps={n_steps}, ent_coef={ent_coef}")
        
        return model
    
    def load_model(
        self,
        model_path: str,
        model_type: str,
        env,
        custom_objects: Optional[Dict[str, Any]] = None
    ) -> Union[SAC, PPO, TD3, A2C]:
        """
        Load a saved model from disk.
        
        Args:
            model_path (str): Path to the saved model
            model_type (str): Type of the model ('sac', 'td3', 'ppo', 'a2c')
            env: Environment to use with the model
            custom_objects (Dict[str, Any], optional): Custom objects to load
            
        Returns:
            Union[SAC, PPO, TD3, A2C]: Loaded model
            
        Raises:
            ValueError: If model_type is invalid
            FileNotFoundError: If model file doesn't exist
        """
        # Map model type to class
        model_classes = {
            'sac': SAC,
            'td3': TD3,
            'ppo': PPO,
            'a2c': A2C
        }
        
        if model_type.lower() not in model_classes:
            raise ValueError(f"Invalid model type '{model_type}'. "
                             f"Valid types are: {list(model_classes.keys())}")
        
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"Model file not found at '{model_path}'")
        
        # Load model
        model_class = model_classes[model_type.lower()]
        model = model_class.load(model_path, env=env, custom_objects=custom_objects)
        
        self.logger.info(f"Loaded {model_type.upper()} model from {model_path}")
        
        return model


class ModelEvaluator:
    """
    Class for evaluating trained models.
    
    This class provides utilities for evaluating model performance
    and generating visualizations of agent behavior.
    
    Attributes:
        logger (logging.Logger): Logger for the evaluator
    """
    
    def __init__(self):
        """Initialize the model evaluator."""
        self.logger = logging.getLogger('ModelEvaluator')
    
    def evaluate_model(
        self,
        model,
        env,
        n_episodes: int = 10,
        deterministic: bool = True,
        render: bool = False
    ) -> Dict[str, float]:
        """
        Evaluate a model on the given environment.
        
        Args:
            model: Trained model to evaluate
            env: Evaluation environment
            n_episodes (int): Number of episodes to evaluate
            deterministic (bool): Whether to use deterministic actions
            render (bool): Whether to render the environment
            
        Returns:
            Dict[str, float]: Dictionary of evaluation metrics
        """
        # Initialize metrics
        episode_rewards = []
        episode_lengths = []
        goal_achieved = 0
        
        # Run evaluation episodes
        for i in range(n_episodes):
            self.logger.info(f"Starting evaluation episode {i+1}/{n_episodes}")
            
            obs = env.reset()
            done = False
            total_reward = 0
            steps = 0
            
            while not done:
                # Get action from model
                action, _states = model.predict(obs, deterministic=deterministic)
                
                # Take step in environment
                obs, reward, done, info = env.step(action)
                
                # Render if requested
                if render:
                    env.render()
                
                # Update metrics
                total_reward += reward
                steps += 1
                
                # Check if goal is achieved (if info available)
                if info.get('is_success', False):
                    goal_achieved += 1
                    break
            
            # Record episode metrics
            episode_rewards.append(total_reward)
            episode_lengths.append(steps)
            
            self.logger.info(f"Episode {i+1} - Reward: {total_reward:.2f}, Steps: {steps}")
        
        # Compile evaluation metrics
        metrics = {
            'mean_reward': np.mean(episode_rewards),
            'std_reward': np.std(episode_rewards),
            'min_reward': np.min(episode_rewards),
            'max_reward': np.max(episode_rewards),
            'mean_length': np.mean(episode_lengths),
            'std_length': np.std(episode_lengths),
            'goal_success_rate': goal_achieved / n_episodes
        }
        
        self.logger.info(f"Evaluation results: {metrics}")
        
        return metrics
    
    def visualize_trajectory(
        self,
        model,
        env,
        output_dir: str = "visualizations",
        max_steps: int = 1000,
        deterministic: bool = True
    ):
        """
        Visualize the agent's trajectory.
        
        Args:
            model: Trained model
            env: Environment
            output_dir (str): Directory to save visualizations
            max_steps (int): Maximum steps per episode
            deterministic (bool): Whether to use deterministic actions
        """
        import matplotlib.pyplot as plt
        from mpl_toolkits.mplot3d import Axes3D
        
        # Create output directory if it doesn't exist
        os.makedirs(output_dir, exist_ok=True)
        
        # Initialize trajectory tracking
        obs = env.reset()
        positions = []
        
        # Get initial position (assuming first 3 elements are x,y,z)
        if hasattr(env, 'goal'):
            goal_position = env.goal
        else:
            goal_position = np.array([0, 0, 0])  # Default goal
        
        # Run episode
        done = False
        steps = 0
        
        while not done and steps < max_steps:
            # Extract position (assuming first 3 elements are x,y,z)
            if hasattr(obs, 'shape') and obs.shape[0] >= 3:
                position = obs[:3]
            else:
                position = np.zeros(3)  # Default position
                
            positions.append(position)
            
            # Get action from model
            action, _states = model.predict(obs, deterministic=deterministic)
            
            # Take step in environment
            obs, reward, done, info = env.step(action)
            
            steps += 1
        
        # Convert to numpy array
        positions = np.array(positions)
        
        # Create 3D plot
        fig = plt.figure(figsize=(10, 8))
        ax = fig.add_subplot(111, projection='3d')
        
        # Plot trajectory
        ax.plot(positions[:, 0], positions[:, 1], positions[:, 2], 'b-', label='Trajectory')
        ax.plot(positions[0, 0], positions[0, 1], positions[0, 2], 'go', label='Start')
        ax.plot(positions[-1, 0], positions[-1, 1], positions[-1, 2], 'ro', label='End')
        
        # Plot goal
        ax.plot(goal_position[0], goal_position[1], goal_position[2], 'yo', markersize=10, label='Goal')
        
        # Set labels and title
        ax.set_xlabel('X')
        ax.set_ylabel('Y')
        ax.set_zlabel('Z')
        ax.set_title('Drone Trajectory')
        ax.legend()
        
        # Save plot
        timestamp = time.strftime("%Y%m%d-%H%M%S")
        filename = os.path.join(output_dir, f"trajectory_{timestamp}.png")
        plt.savefig(filename)
        
        self.logger.info(f"Trajectory visualization saved to {filename}")
        
        plt.close(fig)
