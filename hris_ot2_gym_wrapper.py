import numpy as np
import gymnasium as gym
from gymnasium import spaces
import pybullet as p
import random
import os
import math
import pybullet_data
import time
import matplotlib
# Set non-interactive backend for headless docker environments
matplotlib.use('Agg') 
import matplotlib.pyplot as plt
from clearml import Task, Logger
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.vec_env import DummyVecEnv

# ==============================================================================
# 1. SIMULATION MANAGER (Optimized for FPS)
# ==============================================================================
class Simulation:
    """PyBullet simulation manager for OT-2 robot"""
    def __init__(self, num_agents, render=True, rgb_array=False):
        self.render = render
        self.rgb_array = rgb_array
        mode = p.GUI if render else p.DIRECT
        self.physicsClient = p.connect(mode)
        p.configureDebugVisualizer(p.COV_ENABLE_GUI, 0)
        p.setAdditionalSearchPath(pybullet_data.getDataPath())
        p.setGravity(0, 0, -10)
        
        # Workspace visualization setup
        self.textureId = -1
        if os.path.exists("textures"):
            texture_list = [f for f in os.listdir("textures") if f.endswith('.png')]
            if texture_list:
                random_texture = random.choice(texture_list)
                self.textureId = p.loadTexture(f"textures/{random_texture}")
        
        # Camera setup
        cameraDistance = 1.1 * (math.ceil((num_agents) ** 0.3))
        p.resetDebugVisualizerCamera(cameraDistance, 90, -35, [-0.2, 0.5, 0.1])
        
        self.baseplaneId = p.loadURDF("plane.urdf")
        self.pipette_offset = [0.073, 0.0895, 0.0895]
        self.robotIds = []
        self.specimenIds = []
        self.sphereIds = []
        self.current_frame = None
        self.create_robots(num_agents)

    def create_robots(self, num_agents):
        spacing = 1
        grid_size = math.ceil(num_agents ** 0.5)
        agent_count = 0

        for i in range(grid_size):
            for j in range(grid_size):
                if agent_count >= num_agents: break
                position = [-spacing * i, -spacing * j, 0.03]
                robotId = p.loadURDF("ot_2_simulation_v6.urdf", position, [0, 0, 0, 1], flags=p.URDF_USE_INERTIA_FROM_FILE)
                
                # Specimen setup
                offset = [0.18275-0.00005, 0.163-0.026, 0.057]
                spec_pos = [position[0] + offset[0], position[1] + offset[1], position[2] + offset[2]]
                planeId = p.loadURDF("custom.urdf", spec_pos, p.getQuaternionFromEuler([0, 0, -math.pi/2]))
                p.setCollisionFilterPair(robotId, planeId, -1, -1, enableCollision=0)
                
                if self.textureId >= 0:
                    p.changeVisualShape(planeId, -1, textureUniqueId=self.textureId)

                self.robotIds.append(robotId)
                self.specimenIds.append(planeId)
                agent_count += 1

    def get_pipette_position(self, robotId):
        robot_pos, _ = p.getBasePositionAndOrientation(robotId)
        joint_states = p.getJointStates(robotId, [0, 1, 2])
        x = robot_pos[0] - joint_states[0][0] + self.pipette_offset[0]
        y = robot_pos[1] - joint_states[1][0] + self.pipette_offset[1]
        z = robot_pos[2] + joint_states[2][0] + self.pipette_offset[2]
        return [x, y, z]

    def reset(self, num_agents=1):
        for rId in self.robotIds: p.removeBody(rId)
        for sId in self.specimenIds: p.removeBody(sId)
        self.robotIds, self.specimenIds = [], []
        self.create_robots(num_agents)
        return self.get_states()

    def run(self, actions, num_steps=1):
        for _ in range(num_steps):
            self.apply_actions(actions)
            p.stepSimulation()
            
            # FPS Optimization: No sleep during DIRECT training
            if self.render and not self.rgb_array:
                time.sleep(1./240.)
            
            # Heavy RGB capture only if needed
            if self.rgb_array and _ == num_steps - 1:
                _, _, rgbImg, _, _ = p.getCameraImage(320, 240)
                self.current_frame = rgbImg
                
        return self.get_states()

    def apply_actions(self, actions):
        for i, robotId in enumerate(self.robotIds):
            p.setJointMotorControl2(robotId, 0, p.VELOCITY_CONTROL, targetVelocity=-actions[i][0], force=500)
            p.setJointMotorControl2(robotId, 1, p.VELOCITY_CONTROL, targetVelocity=-actions[i][1], force=500)
            p.setJointMotorControl2(robotId, 2, p.VELOCITY_CONTROL, targetVelocity=actions[i][2], force=800)

    def get_states(self):
        states = {}
        for robotId in self.robotIds:
            states[f'robotId_{robotId}'] = {"pipette_position": self.get_pipette_position(robotId)}
        return states

    def close(self):
        if p.isConnected(): p.disconnect()

# ==============================================================================
# 2. GYM ENVIRONMENT (Metric & Reward Fixed)
# ==============================================================================
class OT2Env(gym.Env):
    def __init__(self, render=False):
        super().__init__()
        self.render_mode = "human" if render else "rgb_array"
        self.sim = Simulation(num_agents=1, render=render, rgb_array=(self.render_mode == "rgb_array"))
        self.action_space = spaces.Box(low=-1.0, high=1.0, shape=(3,), dtype=np.float32)
        self.observation_space = spaces.Box(low=-1.0, high=1.0, shape=(10,), dtype=np.float32)
        
        self.workspace_low = np.array([-0.1871, -0.1706, 0.1195], dtype=np.float32)
        self.workspace_high = np.array([0.2532, 0.2197, 0.2897], dtype=np.float32)
        self.base_position = np.array([0.0, 0.0, 0.03], dtype=np.float32)
        self.pipette_offset = np.array([0.073, 0.0895, 0.0895], dtype=np.float32)
        
        self.goal_pos = np.zeros(3)
        self.steps = 0
        self.max_steps = 1000
        self.prev_dist = 0.0
        self.prev_position = None

    def _normalize_pos(self, pos):
        return 2.0 * (pos - self.workspace_low) / (self.workspace_high - self.workspace_low) - 1.0

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self.steps = 0
        self.prev_position = None
        self.goal_pos = np.clip(self.np_random.uniform(self.workspace_low, self.workspace_high), self.workspace_low, self.workspace_high)
        
        state_dict = self.sim.reset(num_agents=1)
        robot_id = self.sim.robotIds[0]
        
        # Get start position
        curr_pos = np.array(state_dict[f'robotId_{robot_id}']['pipette_position'], dtype=np.float32)
        self.prev_dist = np.linalg.norm(curr_pos - self.goal_pos)
        self.prev_position = curr_pos.copy()
        
        return self._create_obs(curr_pos), {"goal": self.goal_pos}

    def _create_obs(self, pos):
        dist = np.linalg.norm(pos - self.goal_pos)
        norm_dist = np.clip(dist / np.linalg.norm(self.workspace_high - self.workspace_low), 0, 1)
        velocity = (pos - self.prev_position) * 240.0 if self.prev_position is not None else np.zeros(3)
        self.prev_position = pos.copy()
        
        return np.concatenate([
            self._normalize_pos(pos),
            self._normalize_pos(self.goal_pos),
            np.clip(velocity / 0.1, -1, 1),
            [norm_dist * 2 - 1]
        ]).astype(np.float32)

    def step(self, action):
        self.steps += 1
        # Adaptive velocity scaling
        dist_to_goal = self.prev_dist
        max_vel = 0.08 * min(1.0, max(0.05, dist_to_goal / 0.05))
        velocity = np.clip(action, -1.0, 1.0) * max_vel
        
        self.sim.run([[float(velocity[0]), float(velocity[1]), float(velocity[2])]])
        
        state_dict = self.sim.get_states()
        new_pos = np.array(state_dict[next(iter(state_dict))]['pipette_position'], dtype=np.float32)
        curr_dist = np.linalg.norm(new_pos - self.goal_pos)
        
        # Reward Shaping
        reward = (self.prev_dist - curr_dist) * 2000.0
        reward -= 0.01 # Time penalty
        
        terminated = False
        if curr_dist < 0.001: # Success
            reward += 100.0
            terminated = True
        
        self.prev_dist = curr_dist
        truncated = self.steps >= self.max_steps
        
        return self._create_obs(new_pos), float(reward), terminated, truncated, {
            'distance': float(curr_dist),
            'position': new_pos.copy()
        }

# ==============================================================================
# 3. ADVANCED MONITOR CALLBACK (Metric Visibility Fixed)
# ==============================================================================
class AdvancedMonitorCallback(BaseCallback):
    def __init__(self, check_freq=5000, checkpoint_freq=100000, save_path='./models', verbose=1):
        super().__init__(verbose)
        self.check_freq = check_freq
        self.checkpoint_freq = checkpoint_freq
        self.save_path = save_path
        self.best_precision = np.inf
        self.precision_buffer = []
        self.success_buffer = []
        self.current_trajectory = []
        os.makedirs(save_path, exist_ok=True)

    def _init_callback(self) -> None:
        self.clearml_logger = Logger.current_logger()

    def _on_step(self) -> bool:
        info = self.locals['infos'][0]
        
        # CAPTURE METRICS IMMEDIATELY ON EPISODE END
        if self.locals['dones'][0]:
            dist_mm = info.get('distance', 1.0) * 1000
            self.precision_buffer.append(dist_mm)
            self.success_buffer.append(1 if dist_mm <= 1.0 else 0)
            
            if dist_mm < self.best_precision * 1000:
                self.best_precision = info.get('distance')
                if self.best_precision < 0.001: self._save_traj()
            
            if len(self.precision_buffer) > 100:
                self.precision_buffer.pop(0)
                self.success_buffer.pop(0)
            self.current_trajectory = []

        if 'position' in info:
            self.current_trajectory.append(info['position'].copy())

        if self.n_calls % self.check_freq == 0:
            avg_p = np.mean(self.precision_buffer) if self.precision_buffer else 0
            sr = np.mean(self.success_buffer) * 100 if self.success_buffer else 0
            if self.clearml_logger:
                self.clearml_logger.report_scalar("Precision Tracker", "Avg Dist (mm)", avg_p, self.n_calls)
                self.clearml_logger.report_scalar("Precision Tracker", "Success Rate (%)", sr, self.n_calls)
                self.clearml_logger.report_scalar("Precision Tracker", "Best (mm)", self.best_precision * 1000, self.n_calls)
            print(f"Step {self.n_calls:,} | SR: {sr:.1f}% | Avg: {avg_p:.2f}mm | Best: {self.best_precision*1000:.2f}mm")

        if self.n_calls % self.checkpoint_freq == 0:
            self.model.save(os.path.join(self.save_path, f"chk_{self.n_calls}.zip"))
            
        return True

    def _save_traj(self):
        if not self.current_trajectory: return
        try:
            traj = np.array(self.current_trajectory)
            fig = plt.figure(); ax = fig.add_subplot(111, projection='3d')
            ax.plot(traj[:,0], traj[:,1], traj[:,2], 'b-'); ax.scatter(traj[-1,0], traj[-1,1], traj[-1,2], c='r')
            path = os.path.join(self.save_path, f"traj_{self.n_calls}.png")
            plt.savefig(path); plt.close()
            if self.clearml_logger: self.clearml_logger.report_image("Best", "3D Plot", self.n_calls, path)
        except: pass

# ==============================================================================
# 4. MAIN EXECUTION
# ==============================================================================
def main():
    task = Task.init(project_name='Mentor Group - Myrthe/Group 1', task_name='hris_Precision_Training_v3', task_type=Task.TaskTypes.training)
    task.set_repo(repo='https://github.com/AndriiRak243703/Y2B25_Task_11.git', branch='hris/rl-training')
    task.set_base_docker('deanis/2023y2b-rl:latest')
    task.set_packages(['tensorboard', 'clearml', 'gymnasium', 'stable-baselines3==2.2.1', 'pybullet==3.2.5', 'matplotlib'])
    task.execute_remotely(queue_name='default', exit_process=True)
    
    env = DummyVecEnv([lambda: OT2Env(render=False)])
    model = PPO("MlpPolicy", env, learning_rate=2.5e-4, n_steps=2048, batch_size=64, n_epochs=10, gamma=0.998, verbose=1, tensorboard_log="./ppo_logs/")
    
    callback = AdvancedMonitorCallback(check_freq=5000)
    model.learn(total_timesteps=5_000_000, callback=callback, progress_bar=True)
    model.save("final_model"); task.upload_artifact("final_model", "final_model.zip")

if __name__ == "__main__":
    main()