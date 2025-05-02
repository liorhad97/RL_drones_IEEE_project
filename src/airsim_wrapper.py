#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
AirSim Environment Wrapper for RL Drone System

This module provides a wrapper for AirSim to make it compatible with
the existing RL drone architecture. It translates between the RL training
interface and AirSim's API.
"""

import os
import time
import math
import numpy as np
import airsim
import gymnasium as gym
from typing import Dict, List, Optional, Union, Tuple, Any
import logging

# Import project utilities
from utils import setup_logging

class AirSimDroneEnv(gym.Env):
    """
    AirSim environment for drone reinforcement learning.
    
    This class implements the gym.Env interface for AirSim drone control,
    allowing the use of reinforcement learning algorithms with AirSim.
    
    Attributes:
        client (airsim.MultirotorClient): AirSim client
        drone_name (str): Name of the drone in AirSim
        observation_space (gym.spaces.Box): Observation space
        action_space (gym.spaces.Box): Action space
        state (dict): Current drone state
        camera_name (str): Name of the camera to use
        image_shape (tuple): Shape of camera images (height, width, channels)
    """
    
    def __init__(
        self,
        ip_address: str = "127.0.0.1",
        drone_name: str = "PX4",
        camera_name: str = "front_center",
        image_shape: Tuple[int, int, int] = (84, 84, 3),
        use_depth: bool = False,
        use_segmentation: bool = False
    ):
        """
        Initialize the AirSim drone environment.
        
        Args:
            ip_address (str): IP address of the AirSim server
            drone_name (str): Name of the drone in AirSim
            camera_name (str): Name of the camera to use
            image_shape (tuple): Shape of camera images (height, width, channels)
            use_depth (bool): Whether to include depth images in observations
            use_segmentation (bool): Whether to include segmentation images in observations
        """
        # Initialize logger
        self.logger = logging.getLogger("AirSimDroneEnv")
        
        # Connect to AirSim
        self.client = airsim.MultirotorClient(ip=ip_address)
        self.client.confirmConnection()
        
        # Environment settings
        self.drone_name = drone_name
        self.camera_name = camera_name
        self.image_shape = image_shape
        self.use_depth = use_depth
        self.use_segmentation = use_segmentation
        
        # Initialize drone
        self.client.enableApiControl(True, self.drone_name)
        self.client.armDisarm(True, self.drone_name)
        
        # Define observation and action spaces
        # Observation: [position_x, position_y, position_z, 
        #               velocity_x, velocity_y, velocity_z,
        #               roll, pitch, yaw,
        #               rollrate, pitchrate, yawrate]
        self.observation_space = gym.spaces.Box(
            low=np.array([-100, -100, -100, -10, -10, -10, -math.pi, -math.pi/2, -math.pi, -1, -1, -1]),
            high=np.array([100, 100, 100, 10, 10, 10, math.pi, math.pi/2, math.pi, 1, 1, 1]),
            dtype=np.float32
        )
        
        # Action: [vx, vy, vz, yaw_rate] - velocity commands in NED frame
        self.action_space = gym.spaces.Box(
            low=np.array([-1, -1, -1, -1]),  # Normalized velocities and yaw rate
            high=np.array([1, 1, 1, 1]),     # Actual values will be scaled
            dtype=np.float32
        )
        
        # State tracking
        self.state = None
        self.last_obs = None
        self.goal_position = np.array([0, 0, -5])  # Default goal (NED frame)
        
        self.logger.info("AirSim environment initialized. Drone ready.")
    
    def _get_observation(self) -> np.ndarray:
        """
        Get current observation from AirSim.
        
        Returns:
            np.ndarray: Current observation
        """
        # Get multirotor state
        drone_state = self.client.getMultirotorState(vehicle_name=self.drone_name)
        
        # Extract position (NED frame)
        position = drone_state.kinematics_estimated.position
        pos = np.array([position.x_val, position.y_val, position.z_val])
        
        # Extract velocity (NED frame)
        velocity = drone_state.kinematics_estimated.linear_velocity
        vel = np.array([velocity.x_val, velocity.y_val, velocity.z_val])
        
        # Extract orientation (roll, pitch, yaw)
        orientation = drone_state.kinematics_estimated.orientation
        q = np.array([orientation.w_val, orientation.x_val, orientation.y_val, orientation.z_val])
        roll, pitch, yaw = self._quaternion_to_euler(q)
        orientation_euler = np.array([roll, pitch, yaw])
        
        # Extract angular velocity
        angular_velocity = drone_state.kinematics_estimated.angular_velocity
        angular_vel = np.array([angular_velocity.x_val, angular_velocity.y_val, angular_velocity.z_val])
        
        # Create observation vector
        obs = np.concatenate([pos, vel, orientation_euler, angular_vel])
        
        # Track state for later use
        self.state = {
            'position': pos,
            'velocity': vel,
            'orientation': orientation_euler,
            'angular_velocity': angular_vel
        }
        
        return obs.astype(np.float32)
    
    def _quaternion_to_euler(self, q: np.ndarray) -> Tuple[float, float, float]:
        """
        Convert quaternion to euler angles (roll, pitch, yaw).
        
        Args:
            q (np.ndarray): Quaternion [w, x, y, z]
            
        Returns:
            Tuple[float, float, float]: Euler angles (roll, pitch, yaw)
        """
        # Extract quaternion components
        w, x, y, z = q
        
        # Roll (x-axis rotation)
        sinr_cosp = 2 * (w * x + y * z)
        cosr_cosp = 1 - 2 * (x * x + y * y)
        roll = math.atan2(sinr_cosp, cosr_cosp)
        
        # Pitch (y-axis rotation)
        sinp = 2 * (w * y - z * x)
        if abs(sinp) >= 1:
            pitch = math.copysign(math.pi / 2, sinp)  # Use 90 degrees if out of range
        else:
            pitch = math.asin(sinp)
        
        # Yaw (z-axis rotation)
        siny_cosp = 2 * (w * z + x * y)
        cosy_cosp = 1 - 2 * (y * y + z * z)
        yaw = math.atan2(siny_cosp, cosy_cosp)
        
        return roll, pitch, yaw
    
    def _scale_action(self, action: np.ndarray) -> np.ndarray:
        """
        Scale normalized action values to actual control values.
        
        Args:
            action (np.ndarray): Normalized action from policy
            
        Returns:
            np.ndarray: Scaled action for AirSim
        """
        # Scale velocity commands (m/s)
        max_vel_xy = 5.0  # Maximum horizontal velocity
        max_vel_z = 2.0   # Maximum vertical velocity
        max_yaw_rate = 1.0  # Maximum yaw rate (rad/s)
        
        # Scale to actual values
        vx = action[0] * max_vel_xy
        vy = action[1] * max_vel_xy
        vz = action[2] * max_vel_z
        yaw_rate = action[3] * max_yaw_rate
        
        return np.array([vx, vy, vz, yaw_rate])
    
    def step(self, action: np.ndarray) -> Tuple[np.ndarray, float, bool, Dict]:
        """
        Take a step in the environment.
        
        Args:
            action (np.ndarray): Action to take
            
        Returns:
            np.ndarray: Observation
            float: Reward
            bool: Done flag
            Dict: Additional information
        """
        # Scale action to actual control values
        scaled_action = self._scale_action(action)
        
        # Apply velocity command
        vx, vy, vz, yaw_rate = scaled_action
        self.client.moveByVelocityAsync(
            vx, vy, vz,
            duration=0.5,  # Duration of the command
            drivetrain=airsim.DrivetrainType.MaxDegreeOfFreedom,
            yaw_mode=airsim.YawMode(is_rate=True, yaw_or_rate=yaw_rate),
            vehicle_name=self.drone_name
        )
        
        # Short sleep to allow physics to update
        time.sleep(0.05)
        
        # Get new observation
        obs = self._get_observation()
        
        # Compute reward
        reward = self._compute_reward()
        
        # Check if episode is done
        done = self._is_done()
        
        # Additional info
        info = {
            'position': self.state['position'],
            'goal_position': self.goal_position,
            'distance_to_goal': np.linalg.norm(self.state['position'] - self.goal_position)
        }
        
        return obs, reward, done, info
    
    def _compute_reward(self) -> float:
        """
        Compute reward based on current state.
        
        Returns:
            float: Reward value
        """
        # Compute distance to goal
        distance_to_goal = np.linalg.norm(self.state['position'] - self.goal_position)
        
        # Base reward is inversely proportional to distance
        distance_reward = 1.0 / (1.0 + distance_reward)
        
        # Penalty for high velocities (encourages smooth flight)
        velocity_penalty = 0.1 * np.linalg.norm(self.state['velocity'])
        
        # Combine rewards
        reward = distance_reward - velocity_penalty
        
        # Bonus for reaching the goal
        if distance_to_goal < 1.0:
            reward += 10.0
        
        return reward
    
    def _is_done(self) -> bool:
        """
        Check if episode is done.
        
        Returns:
            bool: Done flag
        """
        # Check if reached goal
        distance_to_goal = np.linalg.norm(self.state['position'] - self.goal_position)
        reached_goal = distance_to_goal < 1.0
        
        # Check for collision
        collision_info = self.client.simGetCollisionInfo(vehicle_name=self.drone_name)
        collision = collision_info.has_collided
        
        # Check if out of bounds
        position = self.state['position']
        out_of_bounds = (
            abs(position[0]) > 100 or 
            abs(position[1]) > 100 or 
            abs(position[2]) > 100
        )
        
        return reached_goal or collision or out_of_bounds
    
    def reset(self) -> Tuple[np.ndarray, Dict]:
        """
        Reset the environment for a new episode.
        
        Returns:
            np.ndarray: Initial observation
            Dict: Info dictionary
        """
        # Reset drone to initial position
        self.client.reset()
        self.client.enableApiControl(True, self.drone_name)
        self.client.armDisarm(True, self.drone_name)
        
        # Hover in place to stabilize
        self.client.hoverAsync(vehicle_name=self.drone_name)
        time.sleep(0.2)
        
        # Get initial observation
        obs = self._get_observation()
        self.last_obs = obs
        
        # Initial info
        info = {
            'position': self.state['position'],
            'goal_position': self.goal_position,
            'distance_to_goal': np.linalg.norm(self.state['position'] - self.goal_position)
        }
        
        return obs, info
    
    def set_goal(self, goal_position: np.ndarray) -> None:
        """
        Set a new goal position for the drone.
        
        Args:
            goal_position (np.ndarray): Goal position in NED frame
        """
        self.goal_position = goal_position
        self.logger.info(f"New goal set: {goal_position}")
    
    def get_camera_image(self) -> np.ndarray:
        """
        Get RGB image from the drone camera.
        
        Returns:
            np.ndarray: RGB image with shape (height, width, 3)
        """
        # Request RGB image
        responses = self.client.simGetImages([
            airsim.ImageRequest(self.camera_name, airsim.ImageType.Scene, False, False)
        ], vehicle_name=self.drone_name)
        
        if not responses:
            self.logger.warning("No image returned from AirSim")
            return np.zeros(self.image_shape, dtype=np.uint8)
        
        # Extract and process image
        response = responses[0]
        img_rgba = np.frombuffer(response.image_data_uint8, dtype=np.uint8)
        img_rgba = img_rgba.reshape(response.height, response.width, 4)
        
        # Convert RGBA to RGB
        img_rgb = img_rgba[:, :, :3]
        
        # Resize if needed
        if img_rgb.shape[:2] != self.image_shape[:2]:
            import cv2
            img_rgb = cv2.resize(img_rgb, (self.image_shape[1], self.image_shape[0]))
        
        return img_rgb
    
    def get_depth_image(self) -> np.ndarray:
        """
        Get depth image from the drone camera.
        
        Returns:
            np.ndarray: Depth image with shape (height, width)
        """
        if not self.use_depth:
            return None
        
        # Request depth image
        responses = self.client.simGetImages([
            airsim.ImageRequest(self.camera_name, airsim.ImageType.DepthPlanar, True)
        ], vehicle_name=self.drone_name)
        
        if not responses:
            self.logger.warning("No depth image returned from AirSim")
            return None
        
        # Extract and process depth image
        response = responses[0]
        depth = airsim.list_to_2d_float_array(response.image_data_float, response.width, response.height)
        
        # Resize if needed
        if depth.shape != self.image_shape[:2]:
            import cv2
            depth = cv2.resize(depth, (self.image_shape[1], self.image_shape[0]))
        
        return depth
    
    def close(self) -> None:
        """
        Close the environment and connection to AirSim.
        """
        self.client.enableApiControl(False, self.drone_name)
        self.logger.info("AirSim environment closed")

# For testing the wrapper directly
if __name__ == "__main__":
    # Setup logging
    logger = setup_logging(log_level=logging.INFO)
    
    # Create and test environment
    env = AirSimDroneEnv()
    
    # Test reset
    logger.info("Testing reset...")
    obs, info = env.reset()
    logger.info(f"Initial observation shape: {obs.shape}")
    logger.info(f"Initial position: {info['position']}")
    
    # Test random actions
    logger.info("Testing random actions...")
    for i in range(10):
        action = env.action_space.sample()
        obs, reward, done, info = env.step(action)
        logger.info(f"Step {i}, Reward: {reward}, Position: {info['position']}")
        
        if done:
            logger.info("Episode finished early")
            break
    
    # Close environment
    env.close()
