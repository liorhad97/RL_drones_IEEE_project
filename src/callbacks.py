#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Callbacks for the RL Drone System

This module provides various callback implementations for the RL drone system,
including human feedback, goal adjustment, and logging.
"""

import os
import time
import logging
import numpy as np
import matplotlib.pyplot as plt
from typing import Dict, List, Optional, Union, Any, Callable, Type
from stable_baselines3.common.callbacks import BaseCallback
import keyboard

class HumanFeedbackCallback(BaseCallback):
    """
    Callback for incorporating human feedback during training.
    
    This callback periodically checks for keyboard input to get human feedback,
    which is then incorporated into the reward function.
    
    Attributes:
        feedback_frequency (int): How often to check for feedback (in steps)
        feedback_keys (Dict[str, float]): Mapping from keys to feedback values
        feedback_ui_enabled (bool): Whether to show a UI for feedback
        last_feedback_step (int): Last step when feedback was checked
        feedback_window (Optional): Feedback UI window (if enabled)
    """
    
    def __init__(
        self,
        feedback_frequency: int = 20,
        feedback_keys: Dict[str, float] = {'+': 1.0, '-': -1.0, '0': 0.0},
        feedback_ui_enabled: bool = False,
        verbose: int = 0
    ):
        """
        Initialize the human feedback callback.
        
        Args:
            feedback_frequency (int): How often to check for feedback (in steps)
            feedback_keys (Dict[str, float]): Mapping from keys to feedback values
            feedback_ui_enabled (bool): Whether to show a UI for feedback
            verbose (int): Verbosity level
        """
        super().__init__(verbose)
        self.feedback_frequency = feedback_frequency
        self.feedback_keys = feedback_keys
        self.feedback_ui_enabled = feedback_ui_enabled
        self.last_feedback_step = 0
        self.feedback_window = None
        
        self.logger = logging.getLogger('HumanFeedbackCallback')
        
        # Print instructions
        key_instructions = ', '.join([f"'{k}' for {v}" for k, v in feedback_keys.items()])
        print(f"Human feedback enabled. Press {key_instructions} to provide feedback.")
    
    def _init_callback(self) -> None:
        """Initialize the callback when training starts."""
        if self.feedback_ui_enabled:
            try:
                self._setup_feedback_ui()
            except Exception as e:
                self.logger.error(f"Failed to set up feedback UI: {e}")
                self.feedback_ui_enabled = False
    
    def _setup_feedback_ui(self) -> None:
        """Set up a simple UI for feedback (if matplotlib is available)."""
        try:
            import matplotlib.pyplot as plt
            from matplotlib.widgets import Button
            
            self.fig, self.ax = plt.subplots(figsize=(6, 3))
            self.fig.canvas.set_window_title('Human Feedback')
            
            # Hide axes
            self.ax.axis('off')
            
            # Create buttons for feedback
            button_ax = {}
            n_buttons = len(self.feedback_keys)
            button_width = 0.2
            button_spacing = (1.0 - n_buttons * button_width) / (n_buttons + 1)
            
            for i, (key, value) in enumerate(self.feedback_keys.items()):
                left = (i + 1) * button_spacing + i * button_width
                button_ax[key] = plt.axes([left, 0.4, button_width, 0.3])
                
                # Create button with appropriate label
                if value > 0:
                    label = f"Good (+{value})"
                    color = 'lightgreen'
                elif value < 0:
                    label = f"Bad ({value})"
                    color = 'salmon'
                else:
                    label = "Neutral"
                    color = 'lightgray'
                
                # Create button with callback
                button = Button(button_ax[key], label, color=color)
                button.on_clicked(lambda event, val=value: self._provide_feedback(val))
            
            # Text for feedback status
            self.feedback_text = self.ax.text(0.5, 0.8, "Awaiting feedback...",
                                           ha='center', va='center', fontsize=12)
            
            # Show the figure (non-blocking)
            plt.show(block=False)
            
            self.logger.info("Feedback UI initialized")
            
        except ImportError:
            self.logger.warning("Matplotlib not available, disabling feedback UI")
            self.feedback_ui_enabled = False
    
    def _provide_feedback(self, value: float) -> None:
        """
        Provide feedback to the environment.
        
        Args:
            value (float): Feedback value
        """
        # Check if the environment has a provide_human_feedback method
        if hasattr(self.training_env, 'provide_human_feedback'):
            self.training_env.provide_human_feedback(value)
            self.logger.info(f"Provided feedback: {value}")
            
            # Update UI text if available
            if self.feedback_ui_enabled and hasattr(self, 'feedback_text'):
                if value > 0:
                    text = f"Positive feedback: +{value}"
                    color = 'green'
                elif value < 0:
                    text = f"Negative feedback: {value}"
                    color = 'red'
                else:
                    text = "Neutral feedback: 0"
                    color = 'black'
                
                self.feedback_text.set_text(text)
                self.feedback_text.set_color(color)
                
                # Refresh the plot
                if hasattr(self, 'fig'):
                    self.fig.canvas.draw_idle()
                    self.fig.canvas.flush_events()
    
    def _check_keyboard_input(self) -> None:
        """Check for keyboard input and convert to feedback."""
        for key, value in self.feedback_keys.items():
            if keyboard.is_pressed(key):
                self._provide_feedback(value)
                # Sleep briefly to avoid multiple triggers
                time.sleep(0.1)
                return
    
    def _on_step(self) -> bool:
        """
        Called at each step of training.
        
        Returns:
            bool: Whether training should continue
        """
        # Check for feedback at specified frequency
        if self.num_timesteps - self.last_feedback_step >= self.feedback_frequency:
            try:
                self._check_keyboard_input()
            except Exception as e:
                self.logger.error(f"Error checking keyboard input: {e}")
            
            self.last_feedback_step = self.num_timesteps
        
        # Update feedback UI if enabled
        if self.feedback_ui_enabled and hasattr(self, 'fig'):
            try:
                self.fig.canvas.flush_events()
            except:
                pass
        
        return True
    
    def _on_training_end(self) -> None:
        """Called at the end of training."""
        # Close the feedback UI if it exists
        if self.feedback_ui_enabled and hasattr(self, 'fig'):
            try:
                plt.close(self.fig)
            except:
                pass


class GoalAdjustmentCallback(BaseCallback):
    """
    Callback for adjusting goals based on performance.
    
    This callback periodically evaluates the agent's performance and adjusts
    the difficulty of the goal accordingly.
    
    Attributes:
        evaluation_frequency (int): How often to evaluate performance (in steps)
        success_threshold (float): Threshold for considering a goal successful
        difficulty_levels (List[str]): List of difficulty levels
        start_difficulty (str): Initial difficulty level
        current_difficulty (str): Current difficulty level
        successes (int): Number of successful episodes
        episodes (int): Total number of episodes
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
            evaluation_frequency (int): How often to evaluate performance (in steps)
            success_threshold (float): Threshold for considering a goal successful
            difficulty_levels (List[str]): List of difficulty levels
            start_difficulty (str): Initial difficulty level
            verbose (int): Verbosity level
        """
        super().__init__(verbose)
        self.evaluation_frequency = evaluation_frequency
        self.success_threshold = success_threshold
        self.difficulty_levels = difficulty_levels
        self.start_difficulty = start_difficulty
        self.current_difficulty = start_difficulty
        
        # Performance tracking
        self.successes = 0
        self.episodes = 0
        self.last_eval_step = 0
        
        self.logger = logging.getLogger('GoalAdjustmentCallback')
    
    def _init_callback(self) -> None:
        """Initialize the callback when training starts."""
        # Set initial difficulty
        if hasattr(self.training_env, 'set_goal'):
            self.training_env.set_goal(self.start_difficulty)
            self.logger.info(f"Initial goal difficulty set to {self.start_difficulty}")
    
    def _adjust_difficulty(self) -> None:
        """Adjust goal difficulty based on performance."""
        # Calculate success rate
        if self.episodes == 0:
            return
        
        success_rate = self.successes / self.episodes
        self.logger.info(f"Performance evaluation: Success rate = {success_rate:.2f} "
                        f"({self.successes}/{self.episodes})")
        
        # Determine if difficulty should be changed
        current_idx = self.difficulty_levels.index(self.current_difficulty)
        
        if success_rate >= self.success_threshold and current_idx < len(self.difficulty_levels) - 1:
            # Increase difficulty
            new_difficulty = self.difficulty_levels[current_idx + 1]
            self.logger.info(f"Increasing difficulty from {self.current_difficulty} to {new_difficulty}")
            self.current_difficulty = new_difficulty
            
            # Apply new difficulty
            if hasattr(self.training_env, 'set_goal'):
                self.training_env.set_goal(self.current_difficulty)
        
        elif success_rate < self.success_threshold / 2 and current_idx > 0:
            # Decrease difficulty
            new_difficulty = self.difficulty_levels[current_idx - 1]
            self.logger.info(f"Decreasing difficulty from {self.current_difficulty} to {new_difficulty}")
            self.current_difficulty = new_difficulty
            
            # Apply new difficulty
            if hasattr(self.training_env, 'set_goal'):
                self.training_env.set_goal(self.current_difficulty)
        
        # Reset counters
        self.successes = 0
        self.episodes = 0
    
    def _on_step(self) -> bool:
        """
        Called at each step of training.
        
        Returns:
            bool: Whether training should continue
        """
        # Check for episode termination
        done = self.locals.get('dones', [False])[0]
        
        if done:
            # Update episode counter
            self.episodes += 1
            
            # Check if goal was achieved
            info = self.locals.get('infos', [{}])[0]
            if info.get('is_success', False) or info.get('person_found', False):
                self.successes += 1
        
        # Evaluate and adjust difficulty periodically
        if self.num_timesteps - self.last_eval_step >= self.evaluation_frequency and self.episodes > 0:
            self._adjust_difficulty()
            self.last_eval_step = self.num_timesteps
        
        return True


class LoggingCallback(BaseCallback):
    """
    Callback for logging training progress and saving checkpoints.
    
    Attributes:
        log_frequency (int): How often to log progress (in steps)
        log_dir (str): Directory for logs
        save_frequency (int): How often to save checkpoints (in steps)
        last_log_step (int): Last step when progress was logged
        last_save_step (int): Last step when checkpoint was saved
        rewards (List[float]): List of episodic rewards
        episode_lengths (List[int]): List of episode lengths
        current_episode_reward (float): Reward for current episode
        current_episode_length (int): Length of current episode
    """
    
    def __init__(
        self,
        log_frequency: int = 1000,
        log_dir: str = 'logs',
        save_frequency: int = 10000,
        verbose: int = 0
    ):
        """
        Initialize the logging callback.
        
        Args:
            log_frequency (int): How often to log progress (in steps)
            log_dir (str): Directory for logs
            save_frequency (int): How often to save checkpoints (in steps)
            verbose (int): Verbosity level
        """
        super().__init__(verbose)
        self.log_frequency = log_frequency
        self.log_dir = log_dir
        self.save_frequency = save_frequency
        
        # Logging state
        self.last_log_step = 0
        self.last_save_step = 0
        
        # Episode tracking
        self.rewards = []
        self.episode_lengths = []
        self.current_episode_reward = 0
        self.current_episode_length = 0
        
        self.logger = logging.getLogger('LoggingCallback')
        
        # Create log directory if it doesn't exist
        os.makedirs(log_dir, exist_ok=True)
    
    def _init_callback(self) -> None:
        """Initialize the callback when training starts."""
        # Initialize episode tracking
        self.current_episode_reward = 0
        self.current_episode_length = 0
    
    def _log_progress(self) -> None:
        """Log training progress."""
        # Calculate statistics
        timesteps = self.num_timesteps
        episodes = len(self.rewards)
        
        if episodes > 0:
            mean_reward = np.mean(self.rewards[-100:])
            median_reward = np.median(self.rewards[-100:])
            min_reward = np.min(self.rewards[-100:])
            max_reward = np.max(self.rewards[-100:])
            std_reward = np.std(self.rewards[-100:])
            
            mean_length = np.mean(self.episode_lengths[-100:])
            median_length = np.median(self.episode_lengths[-100:])
            
            # Log to console
            self.logger.info(
                f"Timesteps: {timesteps}, Episodes: {episodes}\n"
                f"Mean reward: {mean_reward:.2f}, Median: {median_reward:.2f}\n"
                f"Min: {min_reward:.2f}, Max: {max_reward:.2f}, Std: {std_reward:.2f}\n"
                f"Mean episode length: {mean_length:.2f}, Median: {median_length:.2f}"
            )
            
            # Log to file
            log_file = os.path.join(self.log_dir, 'training_log.csv')
            header = "timestep,episodes,mean_reward,median_reward,min_reward,max_reward,std_reward,mean_length,median_length"
            
            if not os.path.exists(log_file):
                with open(log_file, 'w') as f:
                    f.write(header + '\n')
            
            with open(log_file, 'a') as f:
                f.write(f"{timesteps},{episodes},{mean_reward},{median_reward},{min_reward},"
                       f"{max_reward},{std_reward},{mean_length},{median_length}\n")
            
            # Create a simple reward plot
            self._plot_rewards()
    
    def _plot_rewards(self) -> None:
        """Create and save a plot of rewards."""
        try:
            # Create figure
            fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 8), sharex=True)
            
            # Plot rewards
            episodes = np.arange(1, len(self.rewards) + 1)
            ax1.plot(episodes, self.rewards, 'b-')
            
            # Add smoothed line
            if len(self.rewards) > 10:
                window_size = min(len(self.rewards) // 10, 100)
                smoothed_rewards = np.convolve(
                    self.rewards, np.ones(window_size) / window_size, mode='valid'
                )
                smoothed_episodes = np.arange(window_size, len(self.rewards) + 1)
                ax1.plot(smoothed_episodes, smoothed_rewards, 'r-', linewidth=2)
            
            ax1.set_ylabel('Episode Reward')
            ax1.set_title('Training Progress')
            ax1.grid(True)
            
            # Plot episode lengths
            ax2.plot(episodes, self.episode_lengths, 'g-')
            
            # Add smoothed line
            if len(self.episode_lengths) > 10:
                window_size = min(len(self.episode_lengths) // 10, 100)
                smoothed_lengths = np.convolve(
                    self.episode_lengths, np.ones(window_size) / window_size, mode='valid'
                )
                ax2.plot(smoothed_episodes, smoothed_lengths, 'm-', linewidth=2)
            
            ax2.set_xlabel('Episode')
            ax2.set_ylabel('Episode Length')
            ax2.grid(True)
            
            plt.tight_layout()
            
            # Save plot
            plot_file = os.path.join(self.log_dir, 'reward_plot.png')
            plt.savefig(plot_file)
            plt.close(fig)
            
        except Exception as e:
            self.logger.error(f"Error creating reward plot: {e}")
    
    def _save_checkpoint(self) -> None:
        """Save a checkpoint of the model."""
        checkpoint_dir = os.path.join(self.log_dir, 'checkpoints')
        os.makedirs(checkpoint_dir, exist_ok=True)
        
        timestamp = time.strftime("%Y%m%d-%H%M%S")
        checkpoint_path = os.path.join(
            checkpoint_dir, f"model_{self.num_timesteps}_{timestamp}"
        )
        
        try:
            self.model.save(checkpoint_path)
            self.logger.info(f"Saved checkpoint to {checkpoint_path}")
            
            # Save episode data
            stats_path = os.path.join(checkpoint_dir, f"stats_{timestamp}.npz")
            np.savez(
                stats_path,
                rewards=np.array(self.rewards),
                episode_lengths=np.array(self.episode_lengths),
                timesteps=np.array(self.num_timesteps)
            )
            
        except Exception as e:
            self.logger.error(f"Error saving checkpoint: {e}")
    
    def _on_step(self) -> bool:
        """
        Called at each step of training.
        
        Returns:
            bool: Whether training should continue
        """
        # Update current episode stats
        if len(self.locals.get('rewards', [])) > 0:
            reward = self.locals['rewards'][0]
            self.current_episode_reward += reward
            self.current_episode_length += 1
        
        # Check for episode termination
        done = self.locals.get('dones', [False])[0]
        if done:
            # Record episode stats
            self.rewards.append(self.current_episode_reward)
            self.episode_lengths.append(self.current_episode_length)
            
            # Reset episode stats
            self.current_episode_reward = 0
            self.current_episode_length = 0
        
        # Log progress periodically
        if self.num_timesteps - self.last_log_step >= self.log_frequency:
            self._log_progress()
            self.last_log_step = self.num_timesteps
        
        # Save checkpoint periodically
        if self.num_timesteps - self.last_save_step >= self.save_frequency:
            self._save_checkpoint()
            self.last_save_step = self.num_timesteps
        
        return True


class CombinedCallback(BaseCallback):
    """
    Callback for combining multiple callbacks into one.
    
    This allows for using multiple callbacks together while maintaining
    a clean interface for the training loop.
    
    Attributes:
        callbacks (List[BaseCallback]): List of callbacks to combine
    """
    
    def __init__(self, callbacks: List[BaseCallback], verbose: int = 0):
        """
        Initialize the combined callback.
        
        Args:
            callbacks (List[BaseCallback]): List of callbacks to combine
            verbose (int): Verbosity level
        """
        super().__init__(verbose)
        self.callbacks = callbacks
    
    def _init_callback(self) -> None:
        """Initialize all callbacks."""
        for callback in self.callbacks:
            callback._init_callback()
    
    def _on_training_start(self) -> None:
        """Called at the start of training."""
        for callback in self.callbacks:
            callback.on_training_start(
                locals=self.locals,
                globals=self.globals
            )
    
    def _on_step(self) -> bool:
        """
        Called at each step of training.
        
        Returns:
            bool: Whether training should continue (False if any callback returns False)
        """
        # Update callback locals and globals
        for callback in self.callbacks:
            callback.update_locals(self.locals)
            callback.update_globals(self.globals)
        
        # Call each callback's _on_step method
        continue_training = True
        for callback in self.callbacks:
            result = callback._on_step()
            # If any callback returns False, stop training
            continue_training = continue_training and result
        
        return continue_training
    
    def _on_training_end(self) -> None:
        """Called at the end of training."""
        for callback in self.callbacks:
            callback.on_training_end()
