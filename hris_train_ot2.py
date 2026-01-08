import gymnasium as gym
from gymnasium import spaces
import pybullet as p
import pybullet_data
import numpy as np
import time
from stable_baselines3 import PPO
from stable_baselines3.common.env_checker import check_env
from stable_baselines3.common.callbacks import CheckpointCallback

class RobotEnv(gym.Env):
    def __init__(self, render=False):
        super(RobotEnv, self).__init__()
        
        # 1. Setup PyBullet
        self.render_mode = render
        if self.render_mode:
            p.connect(p.GUI)
        else:
            p.connect(p.DIRECT)  # High FPS mode for training
            
        p.setAdditionalSearchPath(pybullet_data.getDataPath())
        p.setGravity(0, 0, -9.81)
        
        # 2. Define Action and Observation Spaces
        # Action: Velocity control for 6 joints (or however many your URDF has)
        n_actions = 6 
        self.action_space = spaces.Box(low=-1, high=1, shape=(n_actions,), dtype=np.float32)
        
        # Observation: [Relative_Pos (3), Joint_Angles (6), Joint_Velocities (6)] = 15 dims
        # Relative pos is usually enough, but joints help with kinematics
        self.observation_space = spaces.Box(low=-np.inf, high=np.inf, shape=(15,), dtype=np.float32)
        
        # 3. Robot Settings
        self.robot_id = None
        self.ee_link_idx = 6  # CHANGE THIS to your actual end-effector link index
        self.goal_pos = np.array([0.5, 0.0, 0.5]) # Fixed goal for now, or randomize in reset
        self.success_threshold = 0.01  # 1cm accuracy
        self.max_steps = 1000
        self.current_step = 0
        self.prev_dist = 0.0

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        
        p.resetSimulation()
        p.setGravity(0, 0, -9.81)
        
        # Load Robot (Replace with your URDF path)
        plane_id = p.loadURDF("plane.urdf")
        # Ensure you use the correct path to your robot URDF
        # If using a standard robot like Kuka:
        self.robot_id = p.loadURDF("kuka_iiwa/model.urdf", [0, 0, 0], useFixedBase=True)
        
        # Randomize Goal (Optional: helps generalization)
        # self.goal_pos = np.random.uniform([0.3, -0.2, 0.1], [0.6, 0.2, 0.5])
        
        # Visual Marker for Goal
        p.addUserDebugText("X", self.goal_pos, [1, 0, 0], textSize=2)
        
        # Reset Joints to rest position
        num_joints = p.getNumJoints(self.robot_id)
        for i in range(num_joints):
            p.resetJointState(self.robot_id, i, 0)
            
        self.current_step = 0
        
        # Initialize distance tracking
        current_ee_pos = self._get_ee_pos()
        self.prev_dist = np.linalg.norm(self.goal_pos - current_ee_pos)
        
        return self._get_obs(), {}

    def step(self, action):
        self.current_step += 1
        
        # 1. Apply Action (Velocity Control)
        # Scale action to realistic velocity limits (e.g., 0.5 rad/s)
        scaled_action = action * 0.5
        
        # Assuming first 6 joints are the controllable ones
        p.setJointMotorControlArray(
            self.robot_id, 
            range(6), 
            p.VELOCITY_CONTROL, 
            targetVelocities=scaled_action
        )
        
        # Step Simulation
        p.stepSimulation()
        if self.render_mode:
            time.sleep(1./240.)
            
        # 2. Observations & Metrics
        current_ee_pos = self._get_ee_pos()
        dist = np.linalg.norm(self.goal_pos - current_ee_pos)
        
        # 3. REWARD FUNCTION (The Fix)
        # ----------------------------------------------------------------
        # Part A: Progress Reward (The main driver)
        # If (prev - dist) is positive, we moved closer.
        progress = self.prev_dist - dist
        reward = progress * 100.0  # Scale up so 1cm progress = 1.0 reward
        
        # Part B: Existential Penalty (Force speed)
        reward -= 0.05
        
        # Part C: Success Bonus
        terminated = False
        if dist < self.success_threshold:
            reward += 20.0
            terminated = True
            print(f"🎯 Solved! Distance: {dist:.4f}")
            
        # Update metric for next step
        self.prev_dist = dist
        # ----------------------------------------------------------------
        
        # 4. Termination conditions
        truncated = False
        if self.current_step >= self.max_steps:
            truncated = True
            
        info = {"distance": dist}
        
        return self._get_obs(), reward, terminated, truncated, info

    def _get_ee_pos(self):
        state = p.getLinkState(self.robot_id, self.ee_link_idx)
        return np.array(state[0])

    def _get_obs(self):
        # Joint States
        joint_states = p.getJointStates(self.robot_id, range(6))
        joint_angles = np.array([x[0] for x in joint_states])
        joint_vels = np.array([x[1] for x in joint_states])
        
        # Relative Position (Goal - Current)
        ee_pos = self._get_ee_pos()
        rel_pos = self.goal_pos - ee_pos
        
        # Concatenate for full observation
        return np.concatenate([rel_pos, joint_angles, joint_vels], dtype=np.float32)

    def close(self):
        p.disconnect()


if __name__ == "__main__":
    # 1. Initialize Env
    # render=False for high FPS training
    env = RobotEnv(render=False) 
    
    # Optional: Check env sanity
    # check_env(env)

    # 2. Define Model
    # ent_coef=0.01 prevents premature convergence (the jitter fix)
    model = PPO(
        "MlpPolicy", 
        env, 
        verbose=1, 
        tensorboard_log="./ppo_robot_tensorboard/",
        ent_coef=0.01,
        learning_rate=3e-4,
        batch_size=2048,
        gamma=0.99
    )

    # 3. Train
    print("🚀 Training started... (Press Ctrl+C to stop)")
    try:
        model.learn(total_timesteps=1_000_000)
    except KeyboardInterrupt:
        print("Training interrupted. Saving model...")

    # 4. Save
    model.save("ppo_robot_final")
    print("✅ Model saved as ppo_robot_final.zip")
    
    # 5. Test / Visualize
    print("👀 Visualizing result...")
    env.close()
    
    # Re-open in GUI mode for viewing
    test_env = RobotEnv(render=True)
    model = PPO.load("ppo_robot_final")
    
    obs, _ = test_env.reset()
    for _ in range(1000):
        action, _ = model.predict(obs, deterministic=True)
        obs, reward, done, truncated, info = test_env.step(action)
        if done or truncated:
            obs, _ = test_env.reset()
            
    test_env.close()