import gymnasium as gym
from gymnasium import spaces
import numpy as np
from sim_class import Simulation


class OT2Env(gym.Env):
    """
    Custom Gym environment for OT-2 robot pipette positioning task.
    This wrapper converts the PyBullet simulation into a standard RL environment
    that can be used with Stable-Baselines3 and other RL libraries.
    """

    def __init__(self, render=False, max_steps=300, target_threshold=0.001):
        super(OT2Env, self).__init__()
        
        # Store configuration parameters for use throughout the environment
        self.render_mode = render
        self.max_steps = max_steps
        self.target_threshold = target_threshold
        
        # Creating the PyBullet simulation instance with one robot
        self.sim = Simulation(num_agents=1, render=render)
        
        # Defining action space: normalized [-1, 1] for RL algorithms
        self.action_space = spaces.Box(
            low=np.array([-1.0, -1.0, -1.0], dtype=np.float32),
            high=np.array([1.0, 1.0, 1.0], dtype=np.float32),
            dtype=np.float32
        )
        
        # Defining observation space: 6D normalized positions. All positions are normalized to [-1, 1] range for better RL training.
        self.observation_space = spaces.Box(
            low=-1.0,
            high=1.0,
            shape=(6,),
            dtype=np.float32
        )
        
        # OT-2 workspace physical boundaries in meters.
        self.workspace_low = np.array([-0.1871, -0.1706, 0.1700], dtype=np.float32)
        self.workspace_high = np.array([0.2532, 0.2197, 0.2897], dtype=np.float32)
        
        # Episode tracking
        self.steps = 0
        self.goal_position = None
        self.initial_distance = None



    def reset(self, seed=None):
        """Reset environment to initial state with new random goal."""
        if seed is not None:
            np.random.seed(seed)
        
        # Generating random goal within workspace. This ensures the agent learns to reach any valid position, not just one target.
        self.goal_position = np.random.uniform(
            self.workspace_low,
            self.workspace_high
        ).astype(np.float32)
        
        # Reseting simulation and extracting current position of the pipette.
        state_dict = self.sim.reset(num_agents=1)
        current_pos = self._extract_position(state_dict)
        
        # Calculating and storing initial distance for potential reward scaling.
        self.initial_distance = float(np.linalg.norm(current_pos - self.goal_position))
        
        # Building the observation by concatenating normalized current and goal positions. Normalization ensures all values are in [-1, 1] which helps RL training stability.
        observation = np.concatenate([
            self._normalize_position(current_pos),
            self._normalize_position(self.goal_position)
        ], dtype=np.float32)
        
        # Reset step counter
        self.steps = 0
        
        # Verifying observation shape and dtype. These assertions catch bugs early if something goes wrong.
        assert observation.shape == (6,), f"Observation shape is {observation.shape}, expected (6,)"
        assert observation.dtype == np.float32, f"Observation dtype is {observation.dtype}, expected float32"
        
        return observation, {}



    def step(self, action):
        """Execute one step in the environment."""
        # Converting action to float32 array for consistency.
        action = np.asarray(action, dtype=np.float32)
        
        # Scaling normalized action [-1, 1] to actual velocity commands [-2, 2] m/s.
        max_velocity = 2.0
        velocity = action * max_velocity
        
        # Creating full action array with gripper command (0). Converting to list because sim.run() expects this format.
        full_action = [float(velocity[0]), float(velocity[1]), float(velocity[2]), 0.0]

        # Execute the velocity command in the PyBullet simulation for one timestep.
        state_dict = self.sim.run([full_action])
        
        # Extracting current position.
        current_pos = self._extract_position(state_dict)
        
        # Calculating Euclidean distance from current position to goal.
        distance_to_goal = np.linalg.norm(current_pos - self.goal_position)
        
        # Calculating reward based on current distance.
        reward = self._calculate_reward(distance_to_goal)
        
        # Checking if goal reached.
        terminated = bool(distance_to_goal < self.target_threshold)
        
        # Increment step counter to track episode progress.
        self.steps += 1
        
        # Checking if max steps reached. If max steps reached without success, episode ends as timeout.
        truncated = bool(self.steps >= self.max_steps)
        
        # Building new observation with updated current position and same goal
        observation = np.concatenate([
            self._normalize_position(current_pos),
            self._normalize_position(self.goal_position)
        ], dtype=np.float32)
        
        # Verifying observation shape and dtype
        assert observation.shape == (6,), f"Observation shape is {observation.shape}, expected (6,)"
        assert observation.dtype == np.float32, f"Observation dtype is {observation.dtype}, expected float32"
        
        # Create info dictionary with extra data for logging and debugging
        info = {
            'distance_to_goal': float(distance_to_goal),
            'current_position': current_pos.tolist(),
            'goal_position': self.goal_position.tolist()
        }
        
        return observation, reward, terminated, truncated, info



    def _calculate_reward(self, distance_to_goal):
        """
        Calculate reward signal to guide the RL agent's learning.
        
        Components:
        1. Time penalty: -0.1 per step (punish slow movement)
        2. Distance penalty: -10 * distance (punish being far from goal)
        3. Success bonus: +50 (big reward for reaching goal)
        
        Examples:
        - Reach goal in 100 steps: -10 (time) + -0.05 (distance) + 50 (success) = ~40
        - Reach goal in 200 steps: -20 (time) + -0.05 (distance) + 50 (success) = ~30
        - Timeout without reaching: -30 (time) + -0.1 (distance) = -30.1
        """
        # Time penalty: punish every step. Encourages agent to reach goal quickly, not waste time.
        time_penalty = -0.1
        
        # Distance penalty: punish being far from goal (farther = worse).
        distance_penalty = -10.0 * distance_to_goal
        
        # Success bonus: large reward for reaching the goal. Only given when distance < 1mm (target_threshold).
        success_bonus = 50.0 if distance_to_goal < self.target_threshold else 0.0
        
        reward = time_penalty + distance_penalty + success_bonus
        
        return float(reward)
    
    

    def render(self, mode='human'):
        """Render is handled by simulation if render=True in __init__"""
        pass
    

    def close(self):
        """Close the simulation"""
        self.sim.close()
    

    def _extract_position(self, state_dict):
        """Extract pipette position from state dictionary."""
        robotId = list(sorted(state_dict.keys()))[0]
        robot_state = state_dict.get(robotId, {})
        position = np.array(
            robot_state.get('pipette_position', [0.0, 0.0, 0.0]),
            dtype=np.float32
        )
        return position


    def _normalize_position(self, position):
        """Normalize position from workspace bounds to [-1, 1]."""
        normalized = 2.0 * (position - self.workspace_low) / (self.workspace_high - self.workspace_low) - 1.0
        return normalized.astype(np.float32)