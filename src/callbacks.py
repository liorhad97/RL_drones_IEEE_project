#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Callback Modules for RL Drone System

This module implements callbacks for the RL drone system, including:
- Human feedback integration (RLHF component in the architecture diagram)
- Goal modification based on progress
- Logging and monitoring
- Event handling for keyboard/UI input

Callbacks are used to extend the training and evaluation process without
modifying the core training loop.
"""

import os
import time
import numpy as np
import logging
from typing import Dict, List, Optional, Union, Any, Callable
from stable_baselines3.common.callbacks import BaseCallback, EvalCallback
import keyboard

class HumanFeedbackCallback(BaseCallback):
    """
    Callback for incorporating human feedback during training.
    
    This implements the "RLHF" component (blue box) in the architecture diagram.
    It allows human operators to provide feedback that affects the reward function
    and training process.
    
    Attributes:
        feedback_frequency (int): How often to request/check for feedback (in steps)
        feedback_keys (Dict[str, float]): Mapping of keys to feedback values
        feedback_ui_enabled (bool): Whether to use a graphical UI for feedback
        verbose (int): Verbosity level
        logger (logging.Logger): Logger for the callback
    """
    
    def __init__(
        self,
        feedback_frequency: int = 20,
        feedback_keys: Optional[Dict[str, float]] = None,
        feedback_ui_enabled: bool = False,
        verbose: int = 0
    ):
        """
        Initialize the human feedback callback.
        
        Args:
            feedback_frequency (int): How often to check for feedback (in steps)
            feedback_keys (Dict[str, float], optional): Key mappings for feedback values
            feedback_ui_enabled (bool): Whether to use a graphical UI for feedback
            verbose (int): Verbosity level
        """
        super(HumanFeedbackCallback, self).__init__(verbose)
        self.feedback_frequency = feedback_frequency
        
        # Default key mappings if none provided
        if feedback_keys is None:
            self.feedback_keys = {
                '+': 1.0,   # Positive feedback (good behavior)
                '-': -1.0,  # Negative feedback (bad behavior)
                '0': 0.0    # Neutral feedback
            }
        else:
            self.feedback_keys = feedback_keys
            
        self.feedback_ui_enabled = feedback_ui_enabled
        self.last_feedback_time = time.time()
        self.feedback_cooldown = 0.5  # seconds
        
        # Initialize logger
        self.logger = logging.getLogger('HumanFeedbackCallback')
        self.logger.info(f"Initialized with feedback_frequency={feedback_frequency}")
        
        # Register keyboard handlers if not using UI
        if not feedback_ui_enabled:
            self._setup_keyboard_handlers()
    
    def _setup_keyboard_handlers(self):
        """Set up keyboard handlers for feedback input."""
        # Check if keyboard module is available
        try:
            for key in self.feedback_keys:
                keyboard.on_press_key(key, self._keyboard_callback)
            self.logger.info(f"Registered keyboard handlers for keys: {list(self.feedback_keys.keys())}")
        except ImportError:
            self.logger.warning("Keyboard module not available, keyboard feedback disabled")
        except Exception as e:
            self.logger.error(f"Error setting up keyboard handlers: {e}")
    
    def _keyboard_callback(self, e):
        """
        Handle keyboard events for feedback.
        
        Args:
            e: Keyboard event
        """
        # Check cooldown to prevent rapid-fire feedback
        current_time = time.time()
        if current_time - self.last_feedback_time < self.feedback_cooldown:
            return
        
        # Process the key if it's in our feedback keys
        key = e.name if hasattr(e, 'name') else e.char
        if key in self.feedback_keys:
            feedback = self.feedback_keys[key]
            self._provide_feedback(feedback)
            self.last_feedback_time = current_time
    
    def _provide_feedback(self, feedback_value):
        """
        Provide feedback to the environment.
        
        Args:
            feedback_value (float): Feedback value
        """
        if self.verbose > 0:
            sign = "+" if feedback_value > 0 else ("-" if feedback_value < 0 else "0")
            self.logger.info(f"Step {self.num_timesteps}: Human feedback provided: {sign} ({feedback_value})")
        
        # Apply feedback to the training environment
        # VecEnv requires special handling with env_method
        self.training_env.env_method("provide_human_feedback", feedback_value)
    
    def _on_step(self):
        """
        Callback called at each step.
        
        Checks if it's time to request feedback and processes automatic feedback
        for simulation purposes.
        
        Returns:
            bool: Whether to continue training
        """
        # Check if it's time to prompt for feedback
        if self.num_timesteps % self.feedback_frequency == 0:
            # In simulation mode, generate synthetic feedback
            if not self.feedback_ui_enabled and not keyboard:
                # Simulate human feedback (-1 to +1)
                feedback = np.random.uniform(-0.5, 1.0)  # Biased toward positive feedback
                
                # Apply feedback to the environment
                self._provide_feedback(feedback)
            
            # If UI is enabled, it would be handled by the UI event loop
            # Just log a message that feedback is being requested
            elif self.feedback_ui_enabled:
                if self.verbose > 0:
                    self.logger.info(f"Step {self.num_timesteps}: Requesting human feedback via UI")
        
        return True


class GoalAdjustmentCallback(BaseCallback):
    """
    Callback for adjusting goals based on agent performance.
    
    This callback modifies the goal parameters based on the agent's
    performance, implementing the "Goal Definition" component of the
    architecture diagram.
    
    Attributes:
        evaluation_frequency (int): How often to evaluate and adjust goals (in steps)
        success_threshold (float): Performance threshold for goal adjustment
        difficulty_levels (List[str]): Available difficulty levels
        current_difficulty_idx (int): Index of current difficulty level
        verbose (int): Verbosity level
        logger (logging.Logger): Logger for the callback
    """
    
    def __init__(
        self,
        evaluation_frequency: int = 5000,
        success_threshold: float = 0.8,
        difficulty_levels: List[str] = ['easy', 'medium', 'hard'],
        start_difficulty: str = 'easy',
        verbose: int = 0
    ):
        """
        Initialize the goal adjustment callback.
        
        Args:
            evaluation_frequency (int): How often to evaluate and adjust goals (in steps)
            success_threshold (float): Performance threshold for increasing difficulty
            difficulty_levels (List[str]): Available difficulty levels
            start_difficulty (str): Starting difficulty level
            verbose (int): Verbosity level
        """
        super(GoalAdjustmentCallback, self).__init__(verbose)
        self.evaluation_frequency = evaluation_frequency
        self.success_threshold = success_threshold
        self.difficulty_levels = difficulty_levels
        
        # Set initial difficulty
        if start_difficulty in difficulty_levels:
            self.current_difficulty_idx = difficulty_levels.index(start_difficulty)
        else:
            self.current_difficulty_idx = 0
            
        # Performance tracking
        self.episode_rewards = []
        self.episode_successes = []
        self.current_episode_reward = 0
        self.last_reset_step = 0
        
        # Initialize logger
        self.logger = logging.getLogger('GoalAdjustmentCallback')
        self.logger.info(f"Initialized with evaluation_frequency={evaluation_frequency}, "
                         f"start_difficulty={start_difficulty}")
    
    def _on_step(self):
        """
        Callback called at each step.
        
        Tracks episode rewards and checks if it's time to adjust the goal.
        
        Returns:
            bool: Whether to continue training
        """
        # Get info from the environment (assuming vectorized environment)
        info = self.locals.get('infos')[0]
        
        # Track episode rewards
        self.current_episode_reward += self.locals.get('rewards')[0]
        
        # Check for episode end
        done = self.locals.get('dones')[0]
        if done:
            # Record episode statistics
            self.episode_rewards.append(self.current_episode_reward)
            
            # Check for success (if reported by environment)
            success = info.get('is_success', False)
            self.episode_successes.append(float(success))
            
            # Reset tracking
            self.current_episode_reward = 0
            self.last_reset_step = self.num_timesteps
            
            self.logger.debug(f"Episode ended at step {self.num_timesteps}, success: {success}")
        
        # Check if it's time to evaluate and adjust goals
        if (self.num_timesteps % self.evaluation_frequency == 0) and (self.num_timesteps > 0):
            self._adjust_goal_difficulty()
        
        return True
    
    def _adjust_goal_difficulty(self):
        """
        Adjust goal difficulty based on agent performance.
        
        Increases difficulty if agent is performing well, decreases if struggling.
        """
        # Skip if no episodes completed
        if len(self.episode_successes) == 0:
            return
        
        # Calculate success rate over recent episodes
        recent_episodes = min(len(self.episode_successes), 10)  # Last 10 episodes
        success_rate = np.mean(self.episode_successes[-recent_episodes:])
        
        # Calculate average reward
        avg_reward = np.mean(self.episode_rewards[-recent_episodes:])
        
        # Log current performance
        self.logger.info(f"Current performance - Success rate: {success_rate:.2f}, "
                         f"Average reward: {avg_reward:.2f}, "
                         f"Difficulty: {self.difficulty_levels[self.current_difficulty_idx]}")
        
        # Adjust difficulty based on success rate
        if success_rate >= self.success_threshold:
            # Agent is doing well, increase difficulty if not already at max
            if self.current_difficulty_idx < len(self.difficulty_levels) - 1:
                self.current_difficulty_idx += 1
                new_difficulty = self.difficulty_levels[self.current_difficulty_idx]
                self._set_new_goal(new_difficulty)
                self.logger.info(f"Increasing difficulty to {new_difficulty}")
        elif success_rate < 0.3:  # Arbitrary threshold for poor performance
            # Agent is struggling, decrease difficulty if not already at min
            if self.current_difficulty_idx > 0:
                self.current_difficulty_idx -= 1
                new_difficulty = self.difficulty_levels[self.current_difficulty_idx]
                self._set_new_goal(new_difficulty)
                self.logger.info(f"Decreasing difficulty to {new_difficulty}")
    
    def _set_new_goal(self, difficulty):
        """
        Set a new goal with the specified difficulty.
        
        Args:
            difficulty (str): New difficulty level
        """
        # Apply to all environments (assuming vectorized environment)
        self.training_env.env_method("set_goal", difficulty)
        
        # Reset performance tracking for the new goal
        self.episode_successes = []
        self.episode_rewards = []


class LoggingCallback(BaseCallback):
    """
    Callback for enhanced logging during training.
    
    This callback provides detailed logging of training metrics,
    environment states, and model performance.
    
    Attributes:
        log_frequency (int): How often to log detailed information (in steps)
        log_dir (str): Directory to save log files
        verbose (int): Verbosity level
        logger (logging.Logger): Logger for the callback
    """
    
    def __init__(
        self,
        log_frequency: int = 1000,
        log_dir: str = "logs",
        save_frequency: int = 10000,
        verbose: int = 0
    ):
        """
        Initialize the logging callback.
        
        Args:
            log_frequency (int): How often to log detailed information (in steps)
            log_dir (str): Directory to save log files
            save_frequency (int): How often to save detailed statistics
            verbose (int): Verbosity level
        """
        super(LoggingCallback, self).__init__(verbose)
        self.log_frequency = log_frequency
        self.log_dir = log_dir
        self.save_frequency = save_frequency
        
        # Ensure log directory exists
        os.makedirs(log_dir, exist_ok=True)
        
        # Tracking variables
        self.episode_rewards = []
        self.current_episode_reward = 0
        self.episode_lengths = []
        self.current_episode_length = 0
        
        # Initialize logger
        self.logger = logging.getLogger('LoggingCallback')
        self.logger.info(f"Initialized with log_frequency={log_frequency}, "
                         f"log_dir={log_dir}")
    
    def _on_step(self):
        """
        Callback called at each step.
        
        Tracks performance metrics and logs them at the specified frequency.
        
        Returns:
            bool: Whether to continue training
        """
        # Update episode tracking
        self.current_episode_reward += self.locals.get('rewards')[0]
        self.current_episode_length += 1
        
        # Check for episode end
        done = self.locals.get('dones')[0]
        if done:
            # Record episode statistics
            self.episode_rewards.append(self.current_episode_reward)
            self.episode_lengths.append(self.current_episode_length)
            
            # Log episode completion
            self.logger.info(f"Episode completed: reward={self.current_episode_reward:.2f}, "
                            f"length={self.current_episode_length}")
            
            # Reset tracking
            self.current_episode_reward = 0
            self.current_episode_length = 0
        
        # Log detailed information at specified frequency
        if (self.num_timesteps % self.log_frequency == 0) and (self.num_timesteps > 0):
            self._log_training_status()
        
        # Save detailed statistics at specified frequency
        if (self.num_timesteps % self.save_frequency == 0) and (self.num_timesteps > 0):
            self._save_training_statistics()
        
        return True
    
    def _log_training_status(self):
        """
        Log detailed information about training status.
        """
        # Skip if no episodes completed
        if len(self.episode_rewards) == 0:
            return
        
        # Calculate statistics over recent episodes
        recent_episodes = min(len(self.episode_rewards), 10)  # Last 10 episodes
        avg_reward = np.mean(self.episode_rewards[-recent_episodes:])
        avg_length = np.mean(self.episode_lengths[-recent_episodes:])
        
        # Get additional metrics from the model
        if hasattr(self.model, 'logger') and hasattr(self.model.logger, 'name_to_value'):
            metrics = self.model.logger.name_to_value
            metrics_str = ", ".join([f"{k}: {v:.4f}" for k, v in metrics.items()])
            
            self.logger.info(f"Step {self.num_timesteps}: Avg reward: {avg_reward:.2f}, "
                            f"Avg length: {avg_length:.2f}, {metrics_str}")
        else:
            self.logger.info(f"Step {self.num_timesteps}: Avg reward: {avg_reward:.2f}, "
                            f"Avg length: {avg_length:.2f}")
    
    def _save_training_statistics(self):
        """
        Save detailed training statistics to disk.
        """
        # Create a statistics dictionary
        stats = {
            'timesteps': self.num_timesteps,
            'episodes_completed': len(self.episode_rewards),
            'episode_rewards': self.episode_rewards,
            'episode_lengths': self.episode_lengths,
            'mean_reward': np.mean(self.episode_rewards) if self.episode_rewards else 0,
            'std_reward': np.std(self.episode_rewards) if self.episode_rewards else 0,
            'mean_length': np.mean(self.episode_lengths) if self.episode_lengths else 0,
            'std_length': np.std(self.episode_lengths) if self.episode_lengths else 0,
        }
        
        # Add model-specific metrics if available
        if hasattr(self.model, 'logger') and hasattr(self.model.logger, 'name_to_value'):
            stats.update(self.model.logger.name_to_value)
        
        # Save to file
        timestamp = time.strftime("%Y%m%d-%H%M%S")
        filename = os.path.join(self.log_dir, f"stats_{self.num_timesteps}_{timestamp}.npz")
        np.savez(filename, **stats)
        
        self.logger.info(f"Saved training statistics to {filename}")


class CombinedCallback(BaseCallback):
    """
    Wrapper to combine multiple callbacks.
    
    This allows multiple callbacks to be used together in a coordinated way.
    
    Attributes:
        callbacks (List[BaseCallback]): List of callbacks to execute
    """
    
    def __init__(self, callbacks: List[BaseCallback]):
        """
        Initialize the combined callback.
        
        Args:
            callbacks (List[BaseCallback]): List of callbacks to execute
        """
        super(CombinedCallback, self).__init__(verbose=0)
        self.callbacks = callbacks
        self.logger = logging.getLogger('CombinedCallback')
        self.logger.info(f"Initialized with {len(callbacks)} callbacks")
    
    def _init_callback(self):
        """Initialize all callbacks."""
        for callback in self.callbacks:
            callback._init_callback()
    
    def _on_step(self):
        """
        Execute all callbacks at each step.
        
        Returns:
            bool: Whether to continue training (False if any callback returns False)
        """
        continue_training = True
        
        for callback in self.callbacks:
            # Update callback locals and globals
            callback.locals = self.locals
            callback.globals = self.globals
            
            # Execute the callback
            result = callback._on_step()
            continue_training = continue_training and result
        
        return continue_training
    
    def _on_training_start(self):
        """Notify all callbacks that training is starting."""
        for callback in self.callbacks:
            callback._on_training_start()
    
    def _on_training_end(self):
        """Notify all callbacks that training is ending."""
        for callback in self.callbacks:
            callback._on_training_end()
