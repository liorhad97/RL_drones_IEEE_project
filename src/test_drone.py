#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Test script for the RL drone system.

This script provides a comprehensive test of the different components
of the RL drone system, including the environment wrapper, replay buffer,
callbacks, and models.
"""

import os
import numpy as np
import unittest
import tempfile
import shutil
from stable_baselines3 import SAC
from stable_baselines3.common.env_util import make_vec_env

from gym_pybullet_drones.envs.HoverAviary import HoverAviary
from gym_pybullet_drones.utils.enums import ObservationType, ActionType

# Import custom modules
from env_wrapper import EnhancedDroneEnv
from replay_buffer import EnhancedReplayBuffer
from callbacks import HumanFeedbackCallback, GoalAdjustmentCallback, LoggingCallback, CombinedCallback
from models import ModelFactory
from utils import setup_logging, ensure_dir, calculate_distance_to_goal

class TestEnhancedDroneEnv(unittest.TestCase):
    """Test cases for the EnhancedDroneEnv class."""
    
    def setUp(self):
        """Set up test environment."""
        # Create a temporary directory for test outputs
        self.test_dir = tempfile.mkdtemp()
        
        # Create base environment
        self.base_env = make_vec_env(
            HoverAviary,
            env_kwargs=dict(
                obs=ObservationType.KIN, 
                act=ActionType.RPM,
                gui=False,  # No GUI for testing
            ),
            n_envs=1
        )
        
        # Create enhanced environment
        self.env = EnhancedDroneEnv(
            self.base_env,
            noise_std=0.01,
            use_lidar=True,
            goal_difficulty='easy'
        )
    
    def tearDown(self):
        """Clean up after tests."""
        # Close environments
        self.env.close()
        
        # Remove temporary directory
        shutil.rmtree(self.test_dir)
    
    def test_initialization(self):
        """Test environment initialization."""
        # Check goal is set
        self.assertIsNotNone(self.env.goal)
        self.assertEqual(len(self.env.goal), 3)  # 3D position
        
        # Check observation space is extended when using lidar
        self.assertTrue(self.env.observation_space.shape[0] > self.base_env.observation_space.shape[0])
    
    def test_reset(self):
        """Test environment reset."""
        # Reset environment
        obs = self.env.reset()
        
        # Check observation shape
        self.assertEqual(len(obs), self.env.observation_space.shape[0])
        
        # Check observation is within bounds
        self.assertTrue(np.all(obs >= self.env.observation_space.low))
        self.assertTrue(np.all(obs <= self.env.observation_space.high))
    
    def test_step(self):
        """Test environment step."""
        # Reset environment
        obs = self.env.reset()
        
        # Take a step with zero action
        action = np.zeros(self.env.action_space.shape)
        next_obs, reward, done, info = self.env.step(action)
        
        # Check observation shape
        self.assertEqual(len(next_obs), self.env.observation_space.shape[0])
        
        # Check reward is a scalar
        self.assertTrue(np.isscalar(reward))
        
        # Check done is a boolean
        self.assertIsInstance(done, bool)
        
        # Check info is a dictionary
        self.assertIsInstance(info, dict)
    
    def test_goal_setting(self):
        """Test goal setting functionality."""
        # Test each difficulty level
        for difficulty in ['easy', 'medium', 'hard']:
            self.env.set_goal(difficulty)
            
            # Check goal was updated
            self.assertIsNotNone(self.env.goal)
            self.assertEqual(len(self.env.goal), 3)
            
            # Reset and check if goal affects reset
            obs = self.env.reset()
            self.assertEqual(len(obs), self.env.observation_space.shape[0])
        
        # Test custom goal
        custom_goal = np.array([1.5, -1.5, 2.0])
        self.env.set_goal('custom', custom_goal=custom_goal)
        self.assertTrue(np.array_equal(self.env.goal, custom_goal))
    
    def test_human_feedback(self):
        """Test human feedback functionality."""
        # Reset environment
        obs = self.env.reset()
        
        # Provide positive feedback
        self.env.provide_human_feedback(1.0)
        
        # Take a step
        action = np.zeros(self.env.action_space.shape)
        next_obs, reward, done, info = self.env.step(action)
        
        # Check reward includes feedback (should be higher than normal)
        # This is a basic check - the actual implementation may vary
        self.assertGreater(reward, 0.0)
        
        # Check feedback was reset after use
        self.assertIsNone(self.env.human_feedback)


class TestReplayBuffer(unittest.TestCase):
    """Test cases for the EnhancedReplayBuffer class."""
    
    def setUp(self):
        """Set up test environment."""
        # Create base environment
        self.base_env = make_vec_env(
            HoverAviary,
            env_kwargs=dict(
                obs=ObservationType.KIN, 
                act=ActionType.RPM,
                gui=False,  # No GUI for testing
            ),
            n_envs=1
        )
        
        # Get observation and action spaces
        self.observation_space = self.base_env.observation_space
        self.action_space = self.base_env.action_space
        
        # Create replay buffer
        self.buffer = EnhancedReplayBuffer(
            buffer_size=100,
            observation_space=self.observation_space,
            action_space=self.action_space,
            device="cpu",
            n_envs=1,
            optimize_memory_usage=False,
            prioritized_replay=True,
            alpha=0.6,
            beta=0.4,
            feedback_weight=0.5
        )
    
    def test_buffer_initialization(self):
        """Test buffer initialization."""
        # Check buffer size
        self.assertEqual(self.buffer.buffer_size, 100)
        
        # Check prioritized replay is enabled
        self.assertTrue(self.buffer.prioritized_replay)
        
        # Check priorities are initialized
        self.assertEqual(self.buffer.priorities.shape, (100, 1))
        self.assertEqual(self.buffer.max_priority, 1.0)
    
    def test_add_and_sample(self):
        """Test adding experiences and sampling from buffer."""
        # Create dummy experiences
        for i in range(10):
            obs = self.observation_space.sample()
            next_obs = self.observation_space.sample()
            action = self.action_space.sample()
            reward = float(i)
            done = False
            info = {}
            
            # Add experience with human feedback
            human_feedback = np.array([0.5])
            self.buffer.add(obs, next_obs, action, reward, done, [info], human_feedback)
        
        # Sample from buffer
        batch_size = 5
        data, indices, weights = self.buffer.sample(batch_size)
        
        # Check sample shapes
        self.assertEqual(data["observations"].shape[0], batch_size)
        self.assertEqual(data["actions"].shape[0], batch_size)
        self.assertEqual(data["rewards"].shape[0], batch_size)
        self.assertEqual(data["next_observations"].shape[0], batch_size)
        self.assertEqual(data["dones"].shape[0], batch_size)
        self.assertEqual(data["human_feedback"].shape[0], batch_size)
        
        # Check weights for prioritized replay
        self.assertEqual(weights.shape[0], batch_size)
    
    def test_update_priorities(self):
        """Test updating priorities."""
        # Create dummy experiences
        for i in range(10):
            obs = self.observation_space.sample()
            next_obs = self.observation_space.sample()
            action = self.action_space.sample()
            reward = float(i)
            done = False
            info = {}
            
            # Add experience
            self.buffer.add(obs, next_obs, action, reward, done, [info])
        
        # Sample from buffer
        batch_size = 5
        data, indices, weights = self.buffer.sample(batch_size)
        
        # Update priorities
        td_errors = np.array([0.1, 0.2, 0.3, 0.4, 0.5])
        self.buffer.update_priorities(indices, td_errors)
        
        # Check if priorities were updated
        for i, idx in enumerate(indices):
            self.assertAlmostEqual(self.buffer.priorities[idx][0], td_errors[i], delta=1e-5)
        
        # Max priority should be updated
        self.assertAlmostEqual(self.buffer.max_priority, 0.5, delta=1e-5)
    
    def test_statistics(self):
        """Test getting buffer statistics."""
        # Create dummy experiences
        for i in range(10):
            obs = self.observation_space.sample()
            next_obs = self.observation_space.sample()
            action = self.action_space.sample()
            reward = float(i)
            done = False
            info = {}
            
            # Add experience with varying human feedback
            human_feedback = np.array([float(i) / 10.0])
            self.buffer.add(obs, next_obs, action, reward, done, [info], human_feedback)
        
        # Get statistics
        stats = self.buffer.get_statistics()
        
        # Check statistics exist
        self.assertIn("buffer_size", stats)
        self.assertIn("mean_reward", stats)
        self.assertIn("std_reward", stats)
        self.assertIn("mean_human_feedback", stats)
        
        # Check values make sense
        self.assertEqual(stats["buffer_size"], 10)
        self.assertGreaterEqual(stats["mean_reward"], 0.0)
        self.assertGreaterEqual(stats["mean_human_feedback"], 0.0)


class TestCallbacks(unittest.TestCase):
    """Test cases for the callback classes."""
    
    def setUp(self):
        """Set up test environment."""
        # Create a temporary directory for test outputs
        self.test_dir = tempfile.mkdtemp()
        
        # Create base environment
        self.base_env = make_vec_env(
            HoverAviary,
            env_kwargs=dict(
                obs=ObservationType.KIN, 
                act=ActionType.RPM,
                gui=False,  # No GUI for testing
            ),
            n_envs=1
        )
        
        # Create enhanced environment
        self.env = EnhancedDroneEnv(
            self.base_env,
            noise_std=0.01,
            use_lidar=True,
            goal_difficulty='easy'
        )
        
        # Create SAC model
        self.model = SAC(
            policy="MlpPolicy",
            env=self.env,
            learning_rate=0.0003,
            buffer_size=10000,
            batch_size=256,
            tau=0.005,
            gamma=0.99,
            train_freq=1,
            gradient_steps=1,
            verbose=0
        )
    
    def tearDown(self):
        """Clean up after tests."""
        # Close environments
        self.env.close()
        
        # Remove temporary directory
        shutil.rmtree(self.test_dir)
    
    def test_human_feedback_callback(self):
        """Test HumanFeedbackCallback."""
        # Create callback
        callback = HumanFeedbackCallback(
            feedback_frequency=5,
            feedback_keys={'+': 1.0, '-': -1.0, '0': 0.0},
            feedback_ui_enabled=False,
            verbose=1
        )
        
        # Initialize callback
        callback.init_callback(self.model)
        
        # Test _on_step method
        for i in range(20):
            # Create dummy values for locals and globals
            locals_dict = {
                'timesteps': i,
                'dones': [False],
                'infos': [{}],
                'rewards': [0.0]
            }
            callback.locals = locals_dict
            callback.globals = {}
            
            # Call _on_step
            result = callback._on_step()
            
            # Check result
            self.assertTrue(result)
    
    def test_goal_adjustment_callback(self):
        """Test GoalAdjustmentCallback."""
        # Create callback
        callback = GoalAdjustmentCallback(
            evaluation_frequency=5,
            success_threshold=0.8,
            difficulty_levels=['easy', 'medium', 'hard'],
            start_difficulty='easy',
            verbose=1
        )
        
        # Initialize callback
        callback.init_callback(self.model)
        
        # Test _on_step method
        for i in range(20):
            # Create dummy values for locals and globals
            locals_dict = {
                'timesteps': i,
                'dones': [i % 5 == 0],  # Episode ends every 5 steps
                'infos': [{'is_success': i % 10 == 0}],  # Success every 10 steps
                'rewards': [float(i)]
            }
            callback.locals = locals_dict
            callback.globals = {}
            
            # Call _on_step
            result = callback._on_step()
            
            # Check result
            self.assertTrue(result)
    
    def test_logging_callback(self):
        """Test LoggingCallback."""
        # Create callback
        log_dir = os.path.join(self.test_dir, 'logs')
        callback = LoggingCallback(
            log_frequency=5,
            log_dir=log_dir,
            save_frequency=10,
            verbose=1
        )
        
        # Initialize callback
        callback.init_callback(self.model)
        
        # Test _on_step method
        for i in range(20):
            # Create dummy values for locals and globals
            locals_dict = {
                'timesteps': i,
                'dones': [i % 5 == 0],  # Episode ends every 5 steps
                'infos': [{}],
                'rewards': [float(i)]
            }
            callback.locals = locals_dict
            callback.globals = {}
            
            # Call _on_step
            result = callback._on_step()
            
            # Check result
            self.assertTrue(result)
            
            # Check log directory was created
            if i >= 10:  # After save_frequency
                self.assertTrue(os.path.exists(log_dir))
    
    def test_combined_callback(self):
        """Test CombinedCallback."""
        # Create individual callbacks
        human_feedback_callback = HumanFeedbackCallback(feedback_frequency=5, verbose=0)
        goal_adjustment_callback = GoalAdjustmentCallback(evaluation_frequency=5, verbose=0)
        logging_callback = LoggingCallback(log_frequency=5, log_dir=self.test_dir, verbose=0)
        
        # Create combined callback
        combined_callback = CombinedCallback([
            human_feedback_callback,
            goal_adjustment_callback,
            logging_callback
        ])
        
        # Initialize callback
        combined_callback.init_callback(self.model)
        
        # Test _on_step method
        for i in range(10):
            # Create dummy values for locals and globals
            locals_dict = {
                'timesteps': i,
                'dones': [False],
                'infos': [{}],
                'rewards': [0.0]
            }
            combined_callback.locals = locals_dict
            combined_callback.globals = {}
            
            # Call _on_step
            result = combined_callback._on_step()
            
            # Check result
            self.assertTrue(result)


class TestModels(unittest.TestCase):
    """Test cases for the model factory."""
    
    def setUp(self):
        """Set up test environment."""
        # Create a temporary directory for test outputs
        self.test_dir = tempfile.mkdtemp()
        
        # Create base environment
        self.base_env = make_vec_env(
            HoverAviary,
            env_kwargs=dict(
                obs=ObservationType.KIN, 
                act=ActionType.RPM,
                gui=False,  # No GUI for testing
            ),
            n_envs=1
        )
        
        # Create enhanced environment
        self.env = EnhancedDroneEnv(
            self.base_env,
            noise_std=0.01,
            use_lidar=True,
            goal_difficulty='easy'
        )
        
        # Create model factory
        self.model_factory = ModelFactory()
    
    def tearDown(self):
        """Clean up after tests."""
        # Close environments
        self.env.close()
        
        # Remove temporary directory
        shutil.rmtree(self.test_dir)
    
    def test_create_sac_model(self):
        """Test creating a SAC model."""
        # Create model
        model = self.model_factory.create_sac_model(
            env=self.env,
            learning_rate=0.0003,
            buffer_size=10000,
            batch_size=256,
            tau=0.005,
            gamma=0.99,
            train_freq=1,
            gradient_steps=1,
            verbose=0
        )
        
        # Check model type
        self.assertIsInstance(model, SAC)
        
        # Check model parameters
        self.assertEqual(model.learning_rate, 0.0003)
        self.assertEqual(model.buffer_size, 10000)
        self.assertEqual(model.batch_size, 256)
        self.assertEqual(model.tau, 0.005)
        self.assertEqual(model.gamma, 0.99)
    
    def test_save_and_load_model(self):
        """Test saving and loading a model."""
        # Create model
        model = self.model_factory.create_sac_model(
            env=self.env,
            buffer_size=10000,
            verbose=0
        )
        
        # Save model
        model_path = os.path.join(self.test_dir, 'test_model')
        model.save(model_path)
        
        # Check file exists
        self.assertTrue(os.path.exists(model_path + '.zip'))
        
        # Load model
        loaded_model = self.model_factory.load_model(
            model_path,
            'sac',
            self.env
        )
        
        # Check model type
        self.assertIsInstance(loaded_model, SAC)


class TestUtils(unittest.TestCase):
    """Test cases for utility functions."""
    
    def setUp(self):
        """Set up test environment."""
        # Create a temporary directory for test outputs
        self.test_dir = tempfile.mkdtemp()
    
    def tearDown(self):
        """Clean up after tests."""
        # Remove temporary directory
        shutil.rmtree(self.test_dir)
    
    def test_setup_logging(self):
        """Test setup_logging function."""
        # Set up logging
        log_file = os.path.join(self.test_dir, 'test.log')
        logger = setup_logging(
            log_level=logging.INFO,
            log_file=log_file,
            console_output=True
        )
        
        # Check logger
        self.assertIsInstance(logger, logging.Logger)
        
        # Check log file was created
        self.assertTrue(os.path.exists(log_file))
    
    def test_ensure_dir(self):
        """Test ensure_dir function."""
        # Create a nested directory
        nested_dir = os.path.join(self.test_dir, 'a', 'b', 'c')
        
        # Ensure it exists
        ensure_dir(nested_dir)
        
        # Check directory was created
        self.assertTrue(os.path.exists(nested_dir))
    
    def test_calculate_distance_to_goal(self):
        """Test calculate_distance_to_goal function."""
        # Define position and goal
        position = np.array([1.0, 2.0, 3.0])
        goal = np.array([4.0, 5.0, 6.0])
        
        # Calculate distance
        distance = calculate_distance_to_goal(position, goal)
        
        # Check result is a float
        self.assertIsInstance(distance, float)
        
        # Check value is correct (using Euclidean distance)
        expected = np.sqrt((4.0 - 1.0)**2 + (5.0 - 2.0)**2 + (6.0 - 3.0)**2)
        self.assertAlmostEqual(distance, expected, delta=1e-5)


if __name__ == '__main__':
    unittest.main()
