import gymnasium as gym
import numpy as np
from stable_baselines3 import SAC
from stable_baselines3.common.env_util import make_vec_env

from gym_pybullet_drones.envs.HoverAviary import HoverAviary
from gym_pybullet_drones.utils.enums import ObservationType, ActionType

env = make_vec_env(
    HoverAviary,
    env_kwargs=dict( 
        obs=ObservationType.KIN, 
        act=ActionType.RPM,
        gui=True,
    ),
    n_envs=1
)

# Initialize RL algorithm
model = SAC('MlpPolicy', env, verbose=1,buffer_size= 1000000)

# Try to load existing model if it exists
model_path = "hover_drone_PPO_model"
try:
    model = SAC.load(model_path, env=env)
    print(f"Loaded existing model from {model_path}")
except:
    print(f"No existing model found at {model_path}, starting fresh training")

# Train the agent
model.learn(total_timesteps=2)

# Save the trained model (overwriting the previous one)
model.save(model_path)

# Test the trained agent
obs = env.reset()  # In vectorized environments, reset() returns just the observations
for i in range(1000):
    action, _states = model.predict(obs)
    print(action)
    obs, rewards, dones, info = env.step(action)  # Vectorized environments return 4 values
    # Rendering happens automatically with gui=True
