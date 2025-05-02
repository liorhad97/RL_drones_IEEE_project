#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
AirSim Person Finder Environment

This module adapts the PersonFinderEnv to work with AirSim instead of PyBullet.
It integrates the AirSim drone API with the person finding capabilities.
"""

import os
import time
import math
import numpy as np
import cv2
import logging
import gymnasium as gym
from typing import Dict, List, Optional, Union, Tuple, Any

# Import AirSim wrapper
from airsim_wrapper import AirSimDroneEnv

# Import person detection utilities
from person_detection_utils import PersonDetector, TextPersonMatcher, create_target_person_image

# Import utilities
from utils import setup_logging, calculate_distance_to_goal

class AirSimPersonFinderEnv(gym.Wrapper):
    """
    AirSim environment wrapper for person finding missions with drones.
    
    This wrapper adapts the AirSimDroneEnv for person finding tasks with:
    - Goal definitions based on visual or textual descriptions of target persons
    - Perception capabilities for person detection
    - Feature extraction for person matching
    - Appropriate reward shaping for person finding tasks
    
    Attributes:
        use_visual_goal (bool): Whether using visual or textual goal definition
        detection_threshold (float): Confidence threshold for person detection
        match_threshold (float): Threshold for considering a person match
        target_image (np.ndarray): Image of the target person to find
        target_features (np.ndarray): Visual features of the target person
        target_description (str): Textual description of the target person
        person_detector (PersonDetector): Person detection and feature extraction
        text_matcher (TextPersonMatcher): Text-based person matching
        detected_persons (List): Currently detected persons
        best_match_score (float): Current best match score
        person_found (bool): Whether the target person has been found
    """
    
    def __init__(
        self, 
        env,
        use_visual_goal=True,
        detection_threshold=0.5,
        match_threshold=0.7,
        detector_model_path=None,
        feature_extractor_path=None,
        goal_difficulty='easy',
        camera_resolution=(640, 480),
        device='cpu',
        simulate_detection=False
    ):
        """
        Initialize the AirSim person finder environment.
        
        Args:
            env (AirSimDroneEnv): The base environment to wrap
            use_visual_goal (bool): Whether to use visual or textual goal
            detection_threshold (float): Confidence threshold for person detection
            match_threshold (float): Threshold for considering a person match
            detector_model_path (str, optional): Path to person detector model
            feature_extractor_path (str, optional): Path to feature extractor model
            goal_difficulty (str): Difficulty level for navigation
            camera_resolution (tuple): Camera resolution (width, height)
            device (str): Device for running models ('cpu' or 'cuda')
            simulate_detection (bool): Whether to simulate detection instead of using models
        """
        # Initialize the base environment wrapper
        super().__init__(env)
        
        # Initialize logger
        self.logger = logging.getLogger("AirSimPersonFinderEnv")
        
        # Person finding specific attributes
        self.use_visual_goal = use_visual_goal
        self.detection_threshold = detection_threshold
        self.match_threshold = match_threshold
        self.camera_resolution = camera_resolution
        self.device = device
        self.simulate_detection = simulate_detection
        
        # Goal settings based on difficulty
        self.set_difficulty(goal_difficulty)
        
        # Target person information
        self.target_image = None
        self.target_features = None
        self.target_description = None
        
        # Initialize person detection and matching components
        self.person_detector = PersonDetector(
            detection_threshold=detection_threshold,
            detector_model_path=detector_model_path,
            feature_extractor_path=feature_extractor_path,
            device=device,
            simulate_detection=simulate_detection
        )
        
        self.text_matcher = TextPersonMatcher(match_threshold=match_threshold)
        
        # Tracking detected persons
        self.detected_persons = []
        self.best_match_score = 0.0
        self.person_found = False
        self.human_feedback = None
        
        # Extend observation space to include detection info
        self._extend_observation_space()
        
        self.logger.info(f"Initialized AirSimPersonFinderEnv with use_visual_goal={use_visual_goal}, "
                         f"simulate_detection={simulate_detection}")
    
    def _extend_observation_space(self):
        """Extend the observation space to include detection information."""
        # Get the current observation space
        orig_obs_space = self.observation_space
        
        self.logger.info(f"Original observation space shape: {orig_obs_space.shape}")
        
        # Add dimensions for:
        # - person detection flag (0/1)
        # - match score (0-1)
        # - number of persons detected (int)
        
        # Update observation space
        self.observation_space = gym.spaces.Box(
            low=np.append(orig_obs_space.low, [0, 0, 0]),
            high=np.append(orig_obs_space.high, [1, 1, 10]),  # Assume max 10 persons
            dtype=orig_obs_space.dtype
        )
        
        self.logger.info(f"Extended observation space to {self.observation_space.shape}")
    
    def set_difficulty(self, difficulty: str):
        """
        Set navigation difficulty.
        
        Args:
            difficulty (str): Difficulty level ('easy', 'medium', 'hard')
        """
        # Define goals based on difficulty level
        if difficulty == 'easy':
            # Simple position close to starting point
            goal_position = np.array([5.0, 5.0, -3.0])  # NED coordinates
        elif difficulty == 'medium':
            # Medium distance
            goal_position = np.array([20.0, 20.0, -5.0])
        elif difficulty == 'hard':
            # Far position with altitude variation
            goal_position = np.array([50.0, 50.0, -10.0])
        else:
            raise ValueError(f"Unknown difficulty level: {difficulty}")
        
        # Set goal in base environment
        self.env.set_goal(goal_position)
        self.logger.info(f"Set goal position to {goal_position} (difficulty: {difficulty})")
    
    def set_person_goal(self, target_image=None, target_description=None):
        """
        Set the target person to find using either image or text description.
        
        Args:
            target_image (np.ndarray, optional): Image of target person
            target_description (str, optional): Text description of target person
        
        Raises:
            ValueError: If neither image nor description is provided
        """
        if target_image is None and target_description is None:
            raise ValueError("Must provide either target image or description")
        
        # Reset person finding state
        self.person_found = False
        self.best_match_score = 0.0
        
        # Set target person information
        if target_image is not None:
            self.target_image = target_image
            self.use_visual_goal = True
            
            # Extract features from target image
            self.target_features = self.person_detector.extract_features(target_image)
            self.logger.info("Extracted features from target image")
            
            self.logger.info("Set visual person goal")
        else:
            self.target_description = target_description
            self.use_visual_goal = False
            self.target_image = None
            self.target_features = None
            
            self.logger.info(f"Set text description goal: {target_description[:30]}...")
    
    def step(self, action):
        """
        Step the environment forward with person detection.
        
        Args:
            action (np.ndarray): Action to take in the environment
            
        Returns:
            np.ndarray: Processed observation
            float: Reward
            bool: Done flag
            dict: Info dictionary
        """
        # Execute base environment step
        obs, reward, done, info = self.env.step(action)
        
        # Get camera image
        camera_image = self._get_camera_image()
        
        # Detect persons
        if camera_image is not None:
            self.detected_persons = self.person_detector.detect_persons(camera_image)
        else:
            self.detected_persons = []
        
        # Update observation with detection information
        obs = self._add_detection_info(obs)
        
        # Compute person finding reward
        person_reward = self._compute_person_finding_reward()
        
        # Add person reward to the total reward
        total_reward = reward + person_reward
        
        # Add person finding info to info dict
        info['person_found'] = self.person_found
        info['match_score'] = self.best_match_score
        info['num_detections'] = len(self.detected_persons)
        
        # Apply human feedback if available
        if self.human_feedback is not None:
            total_reward += 0.1 * self.human_feedback
            self.human_feedback = None
        
        return obs, total_reward, done, info
    
    def _add_detection_info(self, obs):
        """
        Add person detection information to observation.
        
        Args:
            obs (np.ndarray): Current observation
            
        Returns:
            np.ndarray: Observation with added detection information
        """
        # Calculate best match score among detected persons
        best_score = 0.0
        
        if self.use_visual_goal and self.target_features is not None:
            # Visual matching
            best_match_idx, best_score = self.person_detector.find_best_match(
                self.detected_persons, self.target_features)
        elif not self.use_visual_goal and self.target_description is not None:
            # Text matching
            best_match_idx, best_score = self.text_matcher.find_best_match(
                self.detected_persons, self.target_description)
        
        # Update best match score
        self.best_match_score = best_score
        
        # Check if person is found
        self.person_found = best_score >= self.match_threshold
        
        # Add detection information to observation:
        # - Detection flag: 1 if any person detected, 0 otherwise
        # - Match score: best match score (0-1)
        # - Number of persons: count of detected persons
        detection_flag = 1.0 if len(self.detected_persons) > 0 else 0.0
        match_score = float(self.best_match_score)
        num_persons = float(len(self.detected_persons))
        
        # Append to observation
        extended_obs = np.append(obs, [detection_flag, match_score, num_persons])
        
        return extended_obs
    
    def _compute_person_finding_reward(self):
        """
        Compute reward for person finding task.
        
        Returns:
            float: Person finding reward
        """
        # Base reward for detecting any person
        detection_reward = 0.1 if len(self.detected_persons) > 0 else 0.0
        
        # Reward for match quality
        match_reward = self.best_match_score * 0.5
        
        # Bonus reward for finding the target person
        found_reward = 2.0 if self.person_found else 0.0
        
        # Combine rewards
        total_reward = detection_reward + match_reward + found_reward
        
        # Log reward components at debug level
        self.logger.debug(f"Person rewards - Detection: {detection_reward:.2f}, "
                         f"Match: {match_reward:.2f}, Found: {found_reward:.2f}")
        
        return total_reward
    
    def reset(self, **kwargs):
        """
        Reset the environment and person finding state.
        
        Args:
            **kwargs: Additional arguments to pass to the base environment reset
            
        Returns:
            np.ndarray: Processed observation
            dict: Info dictionary
        """
        # Reset detection state
        self.detected_persons = []
        self.best_match_score = 0.0
        self.person_found = False
        
        # Call base reset
        obs, info = self.env.reset(**kwargs)
        
        # Add detection info to the observation
        extended_obs = self._add_detection_info(obs)
        
        # Return the extended observation and info
        return extended_obs, info
    
    def _get_camera_image(self):
        """
        Get image from AirSim camera.
        
        Returns:
            np.ndarray: BGR camera image
        """
        # Get RGB image from AirSim
        rgb_image = self.env.get_camera_image()
        
        if rgb_image is None:
            return None
        
        # Convert to BGR for OpenCV compatibility
        bgr_image = cv2.cvtColor(rgb_image, cv2.COLOR_RGB2BGR)
        
        # Resize if needed
        if bgr_image.shape[:2] != self.camera_resolution[::-1]:  # Note: resolution is (width, height)
            bgr_image = cv2.resize(bgr_image, self.camera_resolution)
        
        return bgr_image
    
    def provide_human_feedback(self, feedback_value):
        """
        Provide human feedback to the environment.
        
        Args:
            feedback_value (float): Feedback value, typically in the range [-1, 1]
        """
        self.human_feedback = feedback_value
        self.logger.info(f"Received human feedback: {feedback_value}")
    
    def visualize_detection(self):
        """
        Create visualization of current detection state.
        
        Returns:
            np.ndarray: Visualization image
        """
        # Get camera image
        camera_image = self._get_camera_image()
        if camera_image is None:
            return None
        
        # Find best match index
        best_match_idx = -1
        if self.use_visual_goal and self.target_features is not None:
            best_match_idx, _ = self.person_detector.find_best_match(
                self.detected_persons, self.target_features)
        elif not self.use_visual_goal and self.target_description is not None:
            best_match_idx, _ = self.text_matcher.find_best_match(
                self.detected_persons, self.target_description)
        
        # Create visualization
        vis_img = self.person_detector.visualize_detections(
            camera_image, 
            self.detected_persons, 
            self.target_image, 
            best_match_idx
        )
        
        # Add match score and found status
        h, w = vis_img.shape[:2]
        cv2.putText(vis_img, f"Match: {self.best_match_score:.2f}", (10, h-30),
                  cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        
        status = "FOUND!" if self.person_found else "Searching..."
        color = (0, 255, 0) if self.person_found else (0, 0, 255)
        cv2.putText(vis_img, status, (w-150, 30),
                  cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)
        
        return vis_img
