#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Enhanced Replay Buffer

This module implements an enhanced replay buffer for the drone RL system.
It extends the standard Stable Baselines 3 replay buffer with additional
functionality for storing human feedback and prioritized experience replay.

This represents the "Replay Buffer" component (orange cylinder) in the architecture diagram.
"""

import numpy as np
import logging
from typing import Dict, List, Tuple, Union, Optional
from stable_baselines3.common.buffers import ReplayBuffer
from stable_baselines3.common.vec_env import VecNormalize

class EnhancedReplayBuffer(ReplayBuffer):
    """
    Enhanced replay buffer with additional functionality for human feedback and prioritization.
    
    This buffer extends the standard Stable Baselines 3 ReplayBuffer with features for:
    - Storing human feedback alongside experiences
    - Implementing prioritized experience replay
    - Providing statistics and insights about stored experiences
    
    Attributes:
        human_feedback (np.ndarray): Array of human feedback values for each experience
        priorities (np.ndarray): Priority values for each experience (for prioritized replay)
        feedback_weight (float): Weight given to human feedback in prioritization
    """
    
    def __init__(
        self,
        buffer_size: int,
        observation_space,
        action_space,
        device,
        n_envs: int = 1,
        optimize_memory_usage: bool = False,
        prioritized_replay: bool = False,
        alpha: float = 0.6,
        beta: float = 0.4,
        feedback_weight: float = 0.5
    ):
        """
        Initialize the enhanced replay buffer.
        
        Args:
            buffer_size (int): Maximum size of the buffer
            observation_space: Observation space
            action_space: Action space
            device: PyTorch device
            n_envs (int): Number of parallel environments
            optimize_memory_usage (bool): Optimize memory usage
            prioritized_replay (bool): Whether to use prioritized experience replay
            alpha (float): How much prioritization to use (0=none, 1=full)
            beta (float): Importance sampling correction factor
            feedback_weight (float): Weight given to human feedback in prioritization
        """
        super().__init__(
            buffer_size,
            observation_space,
            action_space,
            device,
            n_envs=n_envs,
            optimize_memory_usage=optimize_memory_usage
        )
        
        # Initialize logger
        self.logger = logging.getLogger('EnhancedReplayBuffer')
        
        # Initialize additional buffers
        self.human_feedback = np.zeros((self.buffer_size, self.n_envs), dtype=np.float32)
        
        # Prioritized replay attributes
        self.prioritized_replay = prioritized_replay
        self.alpha = alpha
        self.beta = beta
        self.feedback_weight = feedback_weight
        
        if self.prioritized_replay:
            self.priorities = np.ones((self.buffer_size, self.n_envs), dtype=np.float32)
            self.max_priority = 1.0
            
        self.logger.info(f"Initialized with buffer_size={buffer_size}, "
                        f"prioritized_replay={prioritized_replay}, "
                        f"feedback_weight={feedback_weight}")
    
    def add(
        self,
        obs: np.ndarray,
        next_obs: np.ndarray,
        action: np.ndarray,
        reward: np.ndarray,
        done: np.ndarray,
        infos: List[Dict],
        human_feedback: Optional[np.ndarray] = None
    ) -> None:
        """
        Add an experience to the buffer.
        
        Extends the standard add method to include human feedback.
        
        Args:
            obs (np.ndarray): Observation
            next_obs (np.ndarray): Next observation
            action (np.ndarray): Action
            reward (np.ndarray): Reward
            done (np.ndarray): Done flag
            infos (List[Dict]): Additional information
            human_feedback (np.ndarray, optional): Human feedback for this experience
        """
        # Call parent add method
        super().add(obs, next_obs, action, reward, done, infos)
        
        # Store human feedback if provided
        if human_feedback is not None:
            self.human_feedback[self.pos] = np.array(human_feedback).reshape(-1, self.n_envs)
            self.logger.debug(f"Added experience with human feedback: {human_feedback}")
        else:
            self.human_feedback[self.pos] = np.zeros((1, self.n_envs), dtype=np.float32)
        
        # Set initial priority for new experience (if using prioritized replay)
        if self.prioritized_replay:
            self.priorities[self.pos] = self.max_priority
    
    def sample(
        self,
        batch_size: int,
        env: Optional[VecNormalize] = None
    ) -> Tuple[Dict[str, np.ndarray], np.ndarray, np.ndarray]:
        """
        Sample experiences from the buffer.
        
        For prioritized replay, samples are weighted by their priorities.
        
        Args:
            batch_size (int): Number of samples to draw
            env (VecNormalize, optional): Environment for normalization
            
        Returns:
            Tuple: (data_dict, indices, weights)
                - data_dict: Dictionary of sampled experiences
                - indices: Indices of sampled experiences
                - weights: Importance sampling weights (for prioritized replay)
        """
        if self.prioritized_replay:
            return self._sample_prioritized(batch_size, env)
        else:
            data = super().sample(batch_size, env)
            
            # Add human feedback to the returned data
            upper_bound = self.buffer_size if self.full else self.pos
            indices = np.random.randint(0, upper_bound, size=batch_size)
            
            # Extract the corresponding human feedback
            feedback = self.human_feedback[indices]
            
            # Add to the data dictionary
            if isinstance(data, tuple):
                data_dict = data[0]
                data_dict["human_feedback"] = feedback
                return data_dict, indices, np.ones_like(indices, dtype=np.float32)
            else:
                data["human_feedback"] = feedback
                return data, indices, np.ones_like(indices, dtype=np.float32)
    
    def _sample_prioritized(
        self,
        batch_size: int,
        env: Optional[VecNormalize] = None
    ) -> Tuple[Dict[str, np.ndarray], np.ndarray, np.ndarray]:
        """
        Sample experiences using prioritized experience replay.
        
        Args:
            batch_size (int): Number of samples to draw
            env (VecNormalize, optional): Environment for normalization
            
        Returns:
            Tuple: (data_dict, indices, weights)
                - data_dict: Dictionary of sampled experiences
                - indices: Indices of sampled experiences
                - weights: Importance sampling weights
        """
        upper_bound = self.buffer_size if self.full else self.pos
        
        # Get probabilities from priorities
        if self.full:
            priorities = self.priorities
        else:
            priorities = self.priorities[:self.pos]
        
        # Apply alpha factor and normalize
        probabilities = priorities ** self.alpha
        probabilities /= probabilities.sum()
        
        # Sample indices based on probabilities
        indices = np.random.choice(upper_bound, size=batch_size, p=probabilities.flatten())
        
        # Calculate importance sampling weights
        weights = (upper_bound * probabilities[indices]) ** (-self.beta)
        weights /= weights.max()  # Normalize weights
        
        # Extract experiences using indices
        data = self._get_samples(indices, env)
        
        # Add human feedback to the returned data
        feedback = self.human_feedback[indices]
        data["human_feedback"] = feedback
        
        self.logger.debug(f"Sampled {batch_size} experiences with prioritized replay")
        
        return data, indices, weights
    
    def update_priorities(
        self,
        indices: np.ndarray,
        td_errors: np.ndarray,
        additional_feedback: Optional[np.ndarray] = None
    ) -> None:
        """
        Update priorities based on TD errors and additional feedback.
        
        Args:
            indices (np.ndarray): Indices of experiences to update
            td_errors (np.ndarray): TD errors for each experience
            additional_feedback (np.ndarray, optional): Additional feedback to incorporate
        """
        if not self.prioritized_replay:
            return
        
        # Ensure positive values
        priorities = np.abs(td_errors)
        
        # Incorporate additional feedback if provided
        if additional_feedback is not None:
            priorities = (1.0 - self.feedback_weight) * priorities + self.feedback_weight * np.abs(additional_feedback)
        
        # Ensure no zero priorities
        priorities = np.maximum(priorities, 1e-6)
        
        # Update priorities
        self.priorities[indices] = priorities
        
        # Update max priority
        self.max_priority = max(self.max_priority, priorities.max())
        
        self.logger.debug(f"Updated priorities for {len(indices)} experiences")
    
    def get_statistics(self) -> Dict[str, float]:
        """
        Get statistics about the replay buffer.
        
        Returns:
            Dict[str, float]: Dictionary of statistics
        """
        upper_bound = self.buffer_size if self.full else self.pos
        
        # Get experiences
        rewards = self.rewards[:upper_bound].flatten()
        human_feedback = self.human_feedback[:upper_bound].flatten()
        
        # Calculate statistics
        stats = {
            "buffer_size": upper_bound,
            "mean_reward": rewards.mean(),
            "std_reward": rewards.std(),
            "min_reward": rewards.min(),
            "max_reward": rewards.max(),
            "mean_human_feedback": human_feedback.mean(),
            "std_human_feedback": human_feedback.std(),
        }
        
        if self.prioritized_replay:
            priorities = self.priorities[:upper_bound].flatten()
            stats.update({
                "mean_priority": priorities.mean(),
                "std_priority": priorities.std(),
                "min_priority": priorities.min(),
                "max_priority": priorities.max(),
            })
        
        self.logger.info(f"Buffer statistics: {stats}")
        
        return stats
