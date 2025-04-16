#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Utility Functions for RL Drone System

This module provides utility functions for logging, data processing,
file management, and visualization for the RL drone system.
"""

import os
import sys
import time
import logging
import numpy as np
from typing import Dict, List, Optional, Union, Any, Tuple
import matplotlib.pyplot as plt
from matplotlib.figure import Figure
from mpl_toolkits.mplot3d import Axes3D

def setup_logging(
    log_level: int = logging.INFO,
    log_file: Optional[str] = None,
    console_output: bool = True
) -> logging.Logger:
    """
    Set up logging configuration.
    
    Args:
        log_level (int): Logging level (e.g., logging.INFO, logging.DEBUG)
        log_file (str, optional): Path to log file
        console_output (bool): Whether to output logs to console
        
    Returns:
        logging.Logger: Configured logger
    """
    # Create logger
    logger = logging.getLogger()
    logger.setLevel(log_level)
    
    # Clear existing handlers
    if logger.hasHandlers():
        logger.handlers.clear()
    
    # Create formatters
    file_formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    console_formatter = logging.Formatter('%(levelname)s: %(message)s')
    
    # Add file handler if log file is specified
    if log_file is not None:
        # Ensure directory exists
        log_dir = os.path.dirname(log_file)
        if log_dir and not os.path.exists(log_dir):
            os.makedirs(log_dir, exist_ok=True)
            
        file_handler = logging.FileHandler(log_file)
        file_handler.setFormatter(file_formatter)
        logger.addHandler(file_handler)
    
    # Add console handler if requested
    if console_output:
        console_handler = logging.StreamHandler(sys.stdout)
        console_handler.setFormatter(console_formatter)
        logger.addHandler(console_handler)
    
    logger.info("Logging initialized")
    
    return logger

def ensure_dir(directory: str) -> None:
    """
    Ensure a directory exists, creating it if necessary.
    
    Args:
        directory (str): Directory path
    """
    if directory and not os.path.exists(directory):
        os.makedirs(directory, exist_ok=True)

def create_timestamp() -> str:
    """
    Create a formatted timestamp string.
    
    Returns:
        str: Formatted timestamp
    """
    return time.strftime("%Y%m%d-%H%M%S")

def calculate_distance_to_goal(
    position: np.ndarray,
    goal: np.ndarray
) -> float:
    """
    Calculate Euclidean distance between current position and goal.
    
    Args:
        position (np.ndarray): Current position
        goal (np.ndarray): Goal position
        
    Returns:
        float: Euclidean distance
    """
    return np.linalg.norm(position - goal)

def plot_learning_curve(
    statistics_file: str,
    output_file: Optional[str] = None,
    window_size: int = 10
) -> Figure:
    """
    Plot learning curves from training statistics.
    
    Args:
        statistics_file (str): Path to statistics file (.npz)
        output_file (str, optional): Path to save the plot
        window_size (int): Window size for smoothing
        
    Returns:
        Figure: Matplotlib figure
    """
    # Load statistics
    try:
        stats = np.load(statistics_file, allow_pickle=True)
    except Exception as e:
        logging.error(f"Failed to load statistics file: {e}")
        raise
    
    # Extract data
    episode_rewards = stats['episode_rewards']
    episode_lengths = stats['episode_lengths']
    timesteps = stats['timesteps']
    
    # Smooth data using rolling average
    def smooth(data, window_size):
        if len(data) < window_size:
            return data
        
        smoothed = np.zeros_like(data)
        for i in range(len(data)):
            start = max(0, i - window_size // 2)
            end = min(len(data), i + window_size // 2 + 1)
            smoothed[i] = np.mean(data[start:end])
        
        return smoothed
    
    smoothed_rewards = smooth(episode_rewards, window_size)
    smoothed_lengths = smooth(episode_lengths, window_size)
    
    # Create figure
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 12), sharex=True)
    
    # Plot rewards
    ax1.plot(range(len(episode_rewards)), episode_rewards, 'b-', alpha=0.3)
    ax1.plot(range(len(smoothed_rewards)), smoothed_rewards, 'b-', linewidth=2)
    ax1.set_ylabel('Episode Reward')
    ax1.set_title('Training Performance')
    ax1.grid(True)
    
    # Plot episode lengths
    ax2.plot(range(len(episode_lengths)), episode_lengths, 'r-', alpha=0.3)
    ax2.plot(range(len(smoothed_lengths)), smoothed_lengths, 'r-', linewidth=2)
    ax2.set_xlabel('Episode')
    ax2.set_ylabel('Episode Length')
    ax2.grid(True)
    
    plt.tight_layout()
    
    # Save figure if output file is specified
    if output_file is not None:
        plt.savefig(output_file)
        logging.info(f"Learning curve saved to {output_file}")
    
    return fig

def plot_3d_trajectory(
    positions: List[np.ndarray],
    goal_position: Optional[np.ndarray] = None,
    output_file: Optional[str] = None
) -> Figure:
    """
    Plot a 3D trajectory.
    
    Args:
        positions (List[np.ndarray]): List of positions (x, y, z)
        goal_position (np.ndarray, optional): Goal position
        output_file (str, optional): Path to save the plot
        
    Returns:
        Figure: Matplotlib figure
    """
    # Convert to numpy array
    positions = np.array(positions)
    
    # Create 3D plot
    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection='3d')
    
    # Plot trajectory
    ax.plot(positions[:, 0], positions[:, 1], positions[:, 2], 'b-', label='Trajectory')
    ax.plot(positions[0, 0], positions[0, 1], positions[0, 2], 'go', markersize=8, label='Start')
    ax.plot(positions[-1, 0], positions[-1, 1], positions[-1, 2], 'ro', markersize=8, label='End')
    
    # Plot goal if provided
    if goal_position is not None:
        ax.plot(goal_position[0], goal_position[1], goal_position[2], 'yo', 
                markersize=10, label='Goal')
    
    # Set labels and title
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    ax.set_title('Drone Trajectory')
    ax.legend()
    
    # Add grid
    ax.grid(True)
    
    # Make the plot more visually appealing
    ax.xaxis.pane.fill = False
    ax.yaxis.pane.fill = False
    ax.zaxis.pane.fill = False
    
    # Save figure if output file is specified
    if output_file is not None:
        plt.savefig(output_file)
        logging.info(f"Trajectory plot saved to {output_file}")
    
    return fig

def calculate_velocity_from_positions(
    positions: List[np.ndarray],
    time_step: float = 1.0
) -> List[np.ndarray]:
    """
    Calculate velocities from position data.
    
    Args:
        positions (List[np.ndarray]): List of positions
        time_step (float): Time step between positions
        
    Returns:
        List[np.ndarray]: List of velocities
    """
    velocities = []
    
    # Convert to numpy array
    positions = np.array(positions)
    
    # Calculate velocities using finite differences
    for i in range(1, len(positions)):
        velocity = (positions[i] - positions[i-1]) / time_step
        velocities.append(velocity)
    
    # Add a zero velocity for the first position
    velocities.insert(0, np.zeros_like(positions[0]))
    
    return velocities

def parse_obs_space(obs):
    """
    Parse observation and extract meaningful components.
    
    Args:
        obs (np.ndarray): Observation
        
    Returns:
        Dict[str, np.ndarray]: Dictionary of observation components
    """
    # This is a template that should be adapted to the specific observation structure
    # For the drone environment, we assume the first 3 elements are position
    parsed = {}
    
    if len(obs) >= 3:
        parsed['position'] = obs[:3]
    
    if len(obs) >= 6:
        parsed['velocity'] = obs[3:6]
    
    if len(obs) >= 9:
        parsed['attitude'] = obs[6:9]
    
    if len(obs) >= 12:
        parsed['angular_velocity'] = obs[9:12]
    
    # If lidar data is included, it might be the last element
    if len(obs) > 12:
        parsed['distance_to_goal'] = obs[-1]
    
    return parsed

def angle_between_vectors(v1, v2):
    """
    Calculate the angle between two vectors.
    
    Args:
        v1 (np.ndarray): First vector
        v2 (np.ndarray): Second vector
        
    Returns:
        float: Angle in radians
    """
    v1_norm = np.linalg.norm(v1)
    v2_norm = np.linalg.norm(v2)
    
    if v1_norm == 0 or v2_norm == 0:
        return 0.0
    
    cos_angle = np.dot(v1, v2) / (v1_norm * v2_norm)
    cos_angle = np.clip(cos_angle, -1.0, 1.0)  # Ensure domain of arccos
    
    return np.arccos(cos_angle)

def rotate_vector(vector, quaternion):
    """
    Rotate a vector using a quaternion.
    
    Args:
        vector (np.ndarray): Vector to rotate
        quaternion (np.ndarray): Quaternion (w, x, y, z)
        
    Returns:
        np.ndarray: Rotated vector
    """
    # Ensure quaternion is normalized
    quat_norm = np.linalg.norm(quaternion)
    if quat_norm == 0:
        return vector
    
    normalized_quat = quaternion / quat_norm
    
    # Extract components
    w, x, y, z = normalized_quat
    
    # Compute rotation
    rotated = np.zeros_like(vector)
    
    # Apply quaternion rotation: v' = q * v * q^-1
    # This is a simplified implementation for 3D vectors
    xx = x * x
    yy = y * y
    zz = z * z
    xy = x * y
    xz = x * z
    yz = y * z
    wx = w * x
    wy = w * y
    wz = w * z
    
    rotated[0] = (1 - 2 * (yy + zz)) * vector[0] + 2 * (xy - wz) * vector[1] + 2 * (xz + wy) * vector[2]
    rotated[1] = 2 * (xy + wz) * vector[0] + (1 - 2 * (xx + zz)) * vector[1] + 2 * (yz - wx) * vector[2]
    rotated[2] = 2 * (xz - wy) * vector[0] + 2 * (yz + wx) * vector[1] + (1 - 2 * (xx + yy)) * vector[2]
    
    return rotated

def load_training_config(config_file):
    """
    Load training configuration from file.
    
    Args:
        config_file (str): Path to configuration file
        
    Returns:
        Dict: Configuration dictionary
    """
    import json
    import yaml
    
    # Determine file type from extension
    _, ext = os.path.splitext(config_file)
    
    try:
        if ext.lower() == '.json':
            with open(config_file, 'r') as f:
                config = json.load(f)
        elif ext.lower() in ['.yaml', '.yml']:
            with open(config_file, 'r') as f:
                config = yaml.safe_load(f)
        else:
            raise ValueError(f"Unsupported configuration file format: {ext}")
        
        logging.info(f"Loaded configuration from {config_file}")
        return config
    
    except Exception as e:
        logging.error(f"Failed to load configuration file: {e}")
        raise

def save_training_config(config, output_file):
    """
    Save training configuration to file.
    
    Args:
        config (Dict): Configuration dictionary
        output_file (str): Path to output file
    """
    import json
    import yaml
    
    # Determine file type from extension
    _, ext = os.path.splitext(output_file)
    
    # Ensure directory exists
    ensure_dir(os.path.dirname(output_file))
    
    try:
        if ext.lower() == '.json':
            with open(output_file, 'w') as f:
                json.dump(config, f, indent=4)
        elif ext.lower() in ['.yaml', '.yml']:
            with open(output_file, 'w') as f:
                yaml.dump(config, f, default_flow_style=False)
        else:
            raise ValueError(f"Unsupported configuration file format: {ext}")
        
        logging.info(f"Saved configuration to {output_file}")
    
    except Exception as e:
        logging.error(f"Failed to save configuration file: {e}")
        raise

def gaussian_exploration_noise(action, scale=0.1, noise_decay=0.9999, min_scale=0.01):
    """
    Add Gaussian noise to action for exploration.
    
    Args:
        action (np.ndarray): Action
        scale (float): Noise scale
        noise_decay (float): Decay factor for noise scale
        min_scale (float): Minimum noise scale
        
    Returns:
        Tuple[np.ndarray, float]: Noisy action and updated scale
    """
    # Add noise
    noisy_action = action + np.random.normal(0, scale, size=action.shape)
    
    # Decay noise scale
    updated_scale = max(scale * noise_decay, min_scale)
    
    return noisy_action, updated_scale

def create_checkpoint_path(base_dir, model_name, timestep):
    """
    Create a path for checkpointing models.
    
    Args:
        base_dir (str): Base directory
        model_name (str): Model name
        timestep (int): Current timestep
        
    Returns:
        str: Checkpoint path
    """
    # Ensure directory exists
    ensure_dir(base_dir)
    
    # Create filename
    timestamp = create_timestamp()
    filename = f"{model_name}_{timestep}_{timestamp}"
    
    return os.path.join(base_dir, filename)

def summarize_model_performance(metrics, output_file=None):
    """
    Summarize model performance.
    
    Args:
        metrics (Dict): Metrics dictionary
        output_file (str, optional): Path to output file
        
    Returns:
        str: Performance summary
    """
    # Create summary
    summary = "Model Performance Summary\n"
    summary += "========================\n\n"
    
    # Add metrics
    for key, value in metrics.items():
        if isinstance(value, (int, float)):
            summary += f"{key}: {value:.4f}\n"
        else:
            summary += f"{key}: {value}\n"
    
    # Save to file if specified
    if output_file is not None:
        with open(output_file, 'w') as f:
            f.write(summary)
        logging.info(f"Performance summary saved to {output_file}")
    
    return summary
