import gymnasium as gym
from gymnasium import spaces
import numpy as np
from sim_class import Simulation


class OT2Env(gym.Env):
    """
    Custom Gymnasium environment for OT-2 robot control.
    
    SIMPLIFIED VERSION - Focus on fast, decisive movement to target.
    
    Observation Space: 6D [current_x, current_y, current_z, goal_x, goal_y, goal_z] (normalized)
    Action Space: 3D [x, y, z] velocities normalized to [-1, 1]
    Reward: Time penalty + Distance penalty + High velocity near target penalty + Success bonus (after settling)
    
    Parameters
    ----------
    render : bool
        Whether to render the simulation visually
    max_steps : int
        Maximum steps per episode before truncation (default: 300)
    target_threshold : float
        Distance threshold (meters) for successful goal achievement (default: 0.001 = 1mm)
    """
    
    def __init__(self, render=False, max_steps=300, target_threshold=0.001):
        super(OT2Env, self).__init__()
        
        self.render_mode = render
        self.max_steps = max_steps
        self.target_threshold = target_threshold
        
        # Create simulation
        self.sim = Simulation(num_agents=1, render=render)
        
        # Define action space: normalized [-1, 1] for RL algorithms
        self.action_space = spaces.Box(
            low=np.array([-1.0, -1.0, -1.0], dtype=np.float32),
            high=np.array([1.0, 1.0, 1.0], dtype=np.float32),
            dtype=np.float32
        )
        
        # Define observation space: 6D normalized positions
        self.observation_space = spaces.Box(
            low=-1.0,
            high=1.0,
            shape=(6,),
            dtype=np.float32
        )
        
        # OT-2 workspace bounds (verified from simulation)
        self.workspace_low = np.array([-0.1871, -0.1706, 0.1700], dtype=np.float32)
        self.workspace_high = np.array([0.2532, 0.2197, 0.2897], dtype=np.float32)
        
        # Episode tracking
        self.steps = 0
        self.goal_position = None
        self.initial_distance = None

        # To ensure that agent does not overshoot
        self.settled_threshold = 5
        self.settling_counter = 0
        self.previous_settling_counter = 0
        self.settled = False
    
    def reset(self, seed=None):
        """Reset environment to initial state with new random goal."""
        if seed is not None:
            np.random.seed(seed)
        
        # Generate random goal within workspace
        self.goal_position = np.random.uniform(
            self.workspace_low,
            self.workspace_high
        ).astype(np.float32)
        
        # Reset simulation
        state_dict = self.sim.reset(num_agents=1)
        
        # Set fixed start position with correct Z coordinates
        self.sim.set_start_position(0.073, 0.0895, 0.1700)
        state_dict = self.sim.get_states()

        # Extract current position
        current_pos = self._extract_position(state_dict)
        print(f"Start: {current_pos}, Goal: {self.goal_position}, Distance: {np.linalg.norm(current_pos - self.goal_position):.4f}")
        
        # Store initial distance for reward scaling
        self.initial_distance = float(np.linalg.norm(current_pos - self.goal_position))

        # Restore initial distance for reward scaling
        self.previous_distance = self.initial_distance
        
        # Create normalized observation
        observation = np.concatenate([
            self._normalize_position(current_pos),
            self._normalize_position(self.goal_position)
        ], dtype=np.float32)
        
        # Reset step counter
        self.steps = 0

        # Reset settling parameters
        self.settling_counter = 0
        self.previous_settling_counter = 0
        self.settled = False
        
        # Verify observation shape and dtype
        assert observation.shape == (6,), f"Observation shape is {observation.shape}, expected (6,)"
        assert observation.dtype == np.float32, f"Observation dtype is {observation.dtype}, expected float32"
        
        return observation, {}
    
    def step(self, action):
        """Execute one step in the environment."""
        # Ensure action is float32
        action = np.asarray(action, dtype=np.float32)
        
        # Scale action to velocity range
        max_velocity = 1.0
        velocity = action * max_velocity
        
        # Create full action array with gripper command (0)
        # Convert to list for sim.run() compatibility
        full_action = [float(velocity[0]), float(velocity[1]), float(velocity[2]), 0.0]

        # Execute action in simulation
        state_dict = self.sim.run([full_action])
        
        # Extract current position
        current_pos = self._extract_position(state_dict)
        
        # Calculate distance to goal
        distance_to_goal = np.linalg.norm(current_pos - self.goal_position)
        
        # Settling
        if distance_to_goal <= self.target_threshold:
            self.settling_counter += 1
            if self.settling_counter == self.settled_threshold:
                self.settled = True
        else:
            self.settling_counter = 0

        # Calculate reward
        reward = self._calculate_reward(distance_to_goal, velocity, max_velocity)
        self.previous_distance = distance_to_goal
        self.previous_settling_counter = self.settling_counter

        # Check if goal reached
        terminated = bool(self.settled)
        
        # Increment step counter
        self.steps += 1
        
        # Check if max steps reached
        truncated = bool(self.steps >= self.max_steps)
        
        # Create observation
        observation = np.concatenate([
            self._normalize_position(current_pos),
            self._normalize_position(self.goal_position)
        ], dtype=np.float32)
        
        # Verify observation shape and dtype
        assert observation.shape == (6,), f"Observation shape is {observation.shape}, expected (6,)"
        assert observation.dtype == np.float32, f"Observation dtype is {observation.dtype}, expected float32"
        
        # Info for logging
        info = {
            'distance_to_goal': float(distance_to_goal),
            'current_position': current_pos.tolist(),
            'goal_position': self.goal_position.tolist(),
            'is_success': terminated
        }
        
        return observation, reward, terminated, truncated, info
    
    def _calculate_reward(self, distance_to_goal, velocity, max_velocity):
        
        # Time penalty (only when NOT in settling zone)
        if distance_to_goal > self.target_threshold:
            time_penalty = -0.1
        else:
            time_penalty = 0.0

        # Distance penalty - punish being far from goal
        distance_penalty = -10.0 * distance_to_goal

        # Settling delta bonus
        settling_delta = self.settling_counter - self.previous_settling_counter
        if settling_delta > 0:
            settling_bonus = 3
        elif settling_delta < 0:
            settling_bonus = 3 * settling_delta
        else:
            settling_bonus = 0

        # Success bonus (only when settled)
        success_bonus = 100.0 if self.settled else 0.0
        
        reward = time_penalty + distance_penalty + settling_bonus + success_bonus
        
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
