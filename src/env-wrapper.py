#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Enhanced Drone Environment Wrapper

This module provides an enhanced wrapper for the drone environment, adding:
- Goal definition with different difficulty levels
- Noise simulation and filtering
- Lidar-based state abstraction
- Human feedback integration into rewards

The wrapper follows the architecture diagram by implementing:
- The "Goal Definition" component (green box in diagram)
- The "Noise Filter" component (pink box)
- The "Lidar mapping" component (tan box)
- Integration with the "Replay Buffer" (orange cylinder)
"""

import gymnasium as gym
import numpy as np
import logging

class EnhancedDroneEnv(gym.Wrapper):
    """
    Enhanced environment wrapper for drone reinforcement learning.
    
    This wrapper adds several features to the base environment:
    - Goal definition with different difficulty levels
    - Noise simulation and filtering for sensor readings
    - Lidar-based state abstraction
    - Integration of human feedback into the reward function
    
    Attributes:
        env (gym.Env): The base environment being wrapped
        noise_std (float): Standard deviation of simulated sensor noise
        use_lidar (bool): Whether to use lidar mapping for state abstraction
        goal (np.ndarray): Current goal position
        human_feedback (float): Current human feedback value
        logger (logging.Logger): Logger for the environment
    """
    
    def __init__(self, env, noise_std=0.01, use_lidar=True, goal_difficulty='easy'):
        """
        Initialize the enhanced drone environment.
        
        Args:
            env (gym.Env): The base environment to wrap
            noise_std (float): Standard deviation of simulated sensor noise
            use_lidar (bool): Whether to use lidar mapping for state abstraction
            goal_difficulty (str): Difficulty level for the goal ('easy', 'medium', 'hard')
        """
        super().__init__(env)
        self.env = env
        self.noise_std = noise_std
        self.use_lidar = use_lidar
        self.goal = None
        self.human_feedback = None
        self.last_obs = None
        self.observation_history = []
        self.logger = logging.getLogger('EnhancedDroneEnv')
        
        # Set goal based on difficulty
        self.set_goal(goal_difficulty)
        
        # Log initialization
        self.logger.info(f"Initialized with noise_std={noise_std}, use_lidar={use_lidar}, goal_difficulty={goal_difficulty}")
        self.logger.info(f"Goal set to {self.goal}")
        
        # If using lidar, extend observation space
        if self.use_lidar:
            # Assuming original observation space is Box
            orig_obs_space = self.observation_space
            
            # Add an additional dimension for the distance to goal
            extended_shape = (orig_obs_space.shape[0] + 1,)
            
            # Update observation space
            self.observation_space = gym.spaces.Box(
                low=np.append(orig_obs_space.low, 0),  # Minimum distance is 0
                high=np.append(orig_obs_space.high, np.inf),  # Maximum distance is infinity
                shape=extended_shape,
                dtype=orig_obs_space.dtype
            )
            
            self.logger.info(f"Extended observation space from {orig_obs_space.shape} to {self.observation_space.shape}")
    
    def set_goal(self, difficulty, custom_goal=None):
        """
        Set the goal based on difficulty level or custom coordinates.
        
        Args:
            difficulty (str): Difficulty level ('easy', 'medium', 'hard')
            custom_goal (np.ndarray, optional): Custom goal coordinates
        """
        if custom_goal is not None:
            self.goal = custom_goal
            self.logger.info(f"Custom goal set to {self.goal}")
            return
            
        if difficulty == 'easy':
            # Simple hovering at fixed position
            self.goal = np.array([0.0, 0.0, 1.0])  # x, y, z coordinates
        elif difficulty == 'medium':
            # Hover and move to a specific position
            self.goal = np.array([1.0, 1.0, 1.5])
        elif difficulty == 'hard':
            # Complex trajectory
            self.goal = np.array([2.0, 2.0, 2.0])
        else:
            raise ValueError(f"Unknown difficulty level: {difficulty}")
        
        self.logger.info(f"Goal set to {self.goal} (difficulty: {difficulty})")
    
    def reset(self, **kwargs):
        """
        Reset the environment and apply goal injection.
        
        Args:
            **kwargs: Additional arguments to pass to the base environment reset
            
        Returns:
            np.ndarray: Processed observation
            dict: Info dictionary (if returned by base environment)
        """
        self.logger.debug("Resetting environment")
        
        # Reset base environment
        result = self.env.reset(**kwargs)
        
        # Clear history
        self.observation_history = []
        self.last_obs = None
        
        # Process the result based on the return type (gym vs gymnasium)
        if isinstance(result, tuple):
            obs, info = result
            processed_obs = self._process_obs(obs)
            return processed_obs, info
        else:
            obs = result
            return self._process_obs(obs)
    
    def step(self, action):
        """
        Step the environment forward with noise filtering and lidar mapping.
        
        Args:
            action (np.ndarray): Action to take in the environment
            
        Returns:
            np.ndarray: Processed observation
            float: Reward
            bool: Done flag
            dict: Info dictionary
        """
        # Execute action in the environment
        obs, reward, done, info = self.env.step(action)
        
        # Log raw data at debug level
        self.logger.debug(f"Raw obs: {obs[:3]}, reward: {reward}, done: {done}")
        
        # Apply custom reward based on goal and possibly human feedback
        custom_reward = self._compute_reward(obs, reward)
        
        # Process observation through noise filter and lidar mapping
        processed_obs = self._process_obs(obs)
        
        # Log processed data
        self.logger.debug(f"Processed obs: {processed_obs[:3]}, reward: {custom_reward}")
        
        return processed_obs, custom_reward, done, info
    
    def _process_obs(self, obs):
        """
        Apply noise filtering and lidar mapping to observations.
        
        Args:
            obs (np.ndarray): Raw observation from the environment
            
        Returns:
            np.ndarray: Processed observation
        """
        # Store raw observation in history
        self.observation_history.append(obs.copy())
        if len(self.observation_history) > 10:
            self.observation_history.pop(0)
        
        # Add realistic noise to simulate sensor inaccuracies
        noisy_obs = obs + np.random.normal(0, self.noise_std, size=obs.shape)
        
        # Apply noise filter
        filtered_obs = self._noise_filter(noisy_obs)
        
        # Apply lidar mapping (state abstraction)
        if self.use_lidar:
            return self._lidar_mapping(filtered_obs)
        
        return filtered_obs
    
    def _noise_filter(self, obs):
        """
        Apply noise filtering to observations.
        
        This implements a simple moving average filter. In a real implementation,
        you might use Kalman filtering or other more sophisticated techniques.
        
        Args:
            obs (np.ndarray): Noisy observation
            
        Returns:
            np.ndarray: Filtered observation
        """
        # First observation handling
        if self.last_obs is None:
            self.last_obs = obs
            return obs
        
        # Apply simple low-pass filter (0.8 * current + 0.2 * previous)
        alpha = 0.8
        filtered_obs = alpha * obs + (1 - alpha) * self.last_obs
        
        # Update last observation
        self.last_obs = obs
        
        return filtered_obs
    
    def _lidar_mapping(self, obs):
        """
        Simulate lidar mapping for state abstraction.
        
        In a real implementation, this would process actual lidar data.
        Here we simply calculate the distance to goal as an abstraction.
        
        Args:
            obs (np.ndarray): Filtered observation
            
        Returns:
            np.ndarray: Observation with added state abstractions
        """
        # Extract position information (assuming first 3 elements are x,y,z position)
        position = obs[:3]
        
        # Calculate distance to goal as a simple abstraction
        distance_to_goal = np.linalg.norm(position - self.goal)
        
        # Add this abstracted information to the observation
        abstracted_obs = np.append(obs, distance_to_goal)
        
        # Log abstraction at debug level
        self.logger.debug(f"Position: {position}, Distance to goal: {distance_to_goal}")
        
        return abstracted_obs
    
    def _compute_reward(self, obs, original_reward):
        """
        Compute custom reward based on goal progress and human feedback.
        
        Args:
            obs (np.ndarray): Raw observation
            original_reward (float): Original reward from the environment
            
        Returns:
            float: Combined reward
        """
        # Extract position from observation (assuming first 3 elements are x,y,z)
        position = obs[:3]
        
        # Calculate distance to goal
        distance_to_goal = np.linalg.norm(position - self.goal)
        
        # Reward is inversely proportional to distance (closer = better)
        # Using a smoother function: 1/(1+distance) gives values in [0,1]
        goal_reward = 1.0 / (1.0 + distance_to_goal)
        
        # Incorporate human feedback if available
        human_reward = 0
        if self.human_feedback is not None:
            human_reward = self.human_feedback
            # Reset after using
            self.human_feedback = None  
            self.logger.info(f"Applied human feedback: {human_reward}")
        
        # Combine rewards (weight them as needed)
        combined_reward = 0.5 * original_reward + 0.4 * goal_reward + 0.1 * human_reward
        
        # Log reward breakdown
        self.logger.debug(f"Reward breakdown - Original: {original_reward:.3f}, "
                         f"Goal: {goal_reward:.3f}, Human: {human_reward:.3f}, "
                         f"Combined: {combined_reward:.3f}")
        
        return combined_reward
    
    def provide_human_feedback(self, feedback_value):
        """
        Provide human feedback to the environment.
        
        This method is called by external components (e.g., the human feedback callback)
        to inject feedback into the reward calculation.
        
        Args:
            feedback_value (float): Feedback value, typically in the range [-1, 1]
        """
        self.human_feedback = feedback_value
        self.logger.info(f"Received human feedback: {feedback_value}")
