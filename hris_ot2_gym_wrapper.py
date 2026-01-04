import numpy as np
import gymnasium as gym
from gymnasium import spaces
import pybullet as p
import random
import os
import math
import pybullet_data
import time

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
        
        # Setup workspace visualization
        texture_list = [f for f in os.listdir("textures") if f.endswith('.png')]
        random_texture = random.choice(texture_list) if texture_list else "default.png"
        self.textureId = p.loadTexture(f"textures/{random_texture}") if os.path.exists(f"textures/{random_texture}") else -1
        
        # Camera setup
        cameraDistance = 1.1 * (math.ceil((num_agents) ** 0.3))
        cameraYaw = 90
        cameraPitch = -35
        cameraTargetPosition = [-0.2, -(math.ceil(num_agents ** 0.5) / 2) + 0.5, 0.1]
        p.resetDebugVisualizerCamera(cameraDistance, cameraYaw, cameraPitch, cameraTargetPosition)
        
        self.baseplaneId = p.loadURDF("plane.urdf")
        self.pipette_offset = [0.073, 0.0895, 0.0895]
        self.pipette_positions = {}
        self.sphereIds = []
        self.droplet_positions = {}
        self.current_frame = None
        self.create_robots(num_agents)

    def create_robots(self, num_agents):
        spacing = 1
        grid_size = math.ceil(num_agents ** 0.5)
        self.robotIds = []
        self.specimenIds = []
        agent_count = 0

        for i in range(grid_size):
            for j in range(grid_size):
                if agent_count >= num_agents:
                    break
                    
                position = [-spacing * i, -spacing * j, 0.03]
                robotId = p.loadURDF("ot_2_simulation_v6.urdf", position, [0, 0, 0, 1],
                                   flags=p.URDF_USE_INERTIA_FROM_FILE)
                
                start_position, start_orientation = p.getBasePositionAndOrientation(robotId)
                p.createConstraint(parentBodyUniqueId=robotId,
                                 parentLinkIndex=-1,
                                 childBodyUniqueId=-1,
                                 childLinkIndex=-1,
                                 jointType=p.JOINT_FIXED,
                                 jointAxis=[0, 0, 0],
                                 parentFramePosition=[0, 0, 0],
                                 childFramePosition=start_position,
                                 childFrameOrientation=start_orientation)

                # Create specimen with texture
                offset = [0.18275-0.00005, 0.163-0.026, 0.057]
                position_with_offset = [position[0] + offset[0], position[1] + offset[1], position[2] + offset[2]]
                rotate_90 = p.getQuaternionFromEuler([0, 0, -math.pi/2])
                planeId = p.loadURDF("custom.urdf", position_with_offset, rotate_90)
                p.setCollisionFilterPair(robotId, planeId, -1, -1, enableCollision=0)
                spec_position, spec_orientation = p.getBasePositionAndOrientation(planeId)
                
                p.createConstraint(parentBodyUniqueId=planeId,
                                 parentLinkIndex=-1,
                                 childBodyUniqueId=-1,
                                 childLinkIndex=-1,
                                 jointType=p.JOINT_FIXED,
                                 jointAxis=[0, 0, 0],
                                 parentFramePosition=[0, 0, 0],
                                 childFramePosition=spec_position,
                                 childFrameOrientation=spec_orientation)
                
                if self.textureId >= 0:
                    p.changeVisualShape(planeId, -1, textureUniqueId=self.textureId)

                self.robotIds.append(robotId)
                self.specimenIds.append(planeId)
                self.pipette_positions[f'robotId_{robotId}'] = self.get_pipette_position(robotId)
                agent_count += 1

    def get_pipette_position(self, robotId):
        robot_position = p.getBasePositionAndOrientation(robotId)[0]
        joint_states = p.getJointStates(robotId, [0, 1, 2])
        robot_position = list(robot_position)
        robot_position[0] -= joint_states[0][0]
        robot_position[1] -= joint_states[1][0]
        robot_position[2] += joint_states[2][0]
        
        return [
            robot_position[0] + self.pipette_offset[0],
            robot_position[1] + self.pipette_offset[1],
            robot_position[2] + self.pipette_offset[2]
        ]

    def reset(self, num_agents=1):
        # Cleanup existing objects
        for robotId in self.robotIds:
            p.removeBody(robotId)
        for specimenId in self.specimenIds:
            p.removeBody(specimenId)
        for sphereId in self.sphereIds:
            p.removeBody(sphereId)
            
        # Reset tracking structures
        self.pipette_positions = {}
        self.sphereIds = []
        self.droplet_positions = {}
        self.current_frame = None
        
        # Recreate environment
        self.create_robots(num_agents)
        return self.get_states()

    def run(self, actions, num_steps=1):
        for _ in range(num_steps):
            self.apply_actions(actions)
            p.stepSimulation()
            
            # Contact handling
            for specimenId, robotId in zip(self.specimenIds, self.robotIds):
                self.check_contact(robotId, specimenId)
            
            # RGB capture if needed
            if self.rgb_array:
                width, height, rgbImg, _, _ = p.getCameraImage(
                    width=320, height=240,
                    viewMatrix=p.computeViewMatrix([1, 0, 1], [-0.3, 0, 0], [0, 0, 1]),
                    projectionMatrix=p.computeProjectionMatrixFOV(50, 320/240, 0.1, 100.0)
                )
                self.current_frame = rgbImg
            
            if self.render:
                time.sleep(1./240.)
                
        return self.get_states()

    def apply_actions(self, actions):
        for i, robotId in enumerate(self.robotIds):
            p.setJointMotorControl2(robotId, 0, p.VELOCITY_CONTROL, 
                                  targetVelocity=-actions[i][0], force=500)
            p.setJointMotorControl2(robotId, 1, p.VELOCITY_CONTROL, 
                                  targetVelocity=-actions[i][1], force=500)
            p.setJointMotorControl2(robotId, 2, p.VELOCITY_CONTROL, 
                                  targetVelocity=actions[i][2], force=800)

    def get_states(self):
        states = {}
        for robotId in self.robotIds:
            joint_states = p.getJointStates(robotId, [0, 1, 2])
            robot_position = p.getBasePositionAndOrientation(robotId)[0]
            robot_position = list(robot_position)
            robot_position[0] -= joint_states[0][0]
            robot_position[1] -= joint_states[1][0]
            robot_position[2] += joint_states[2][0]
            
            pipette_position = [
                robot_position[0] + self.pipette_offset[0],
                robot_position[1] + self.pipette_offset[1],
                robot_position[2] + self.pipette_offset[2]
            ]
            
            states[f'robotId_{robotId}'] = {
                "joint_states": {
                    f'joint_{i}': {
                        'position': joint_states[i][0],
                        'velocity': joint_states[i][1]
                    } for i in range(3)
                },
                "robot_position": robot_position,
                "pipette_position": [round(coord, 4) for coord in pipette_position]
            }
        return states

    def check_contact(self, robotId, specimenId):
        for sphereId in self.sphereIds.copy():
            if not p.isConnected():
                return
                
            contact_points = p.getContactPoints(sphereId, specimenId)
            if contact_points:
                sphere_pos, sphere_orient = p.getBasePositionAndOrientation(sphereId)
                p.setCollisionFilterPair(sphereId, specimenId, -1, -1, enableCollision=0)
                p.createConstraint(sphereId, -1, -1, -1, p.JOINT_FIXED, [0,0,0],
                                  [0,0,0], sphere_pos, sphere_orient)
                
                key = f'specimenId_{specimenId}'
                if key not in self.droplet_positions:
                    self.droplet_positions[key] = []
                self.droplet_positions[key].append(sphere_pos)
                
                if sphereId in self.sphereIds:
                    self.sphereIds.remove(sphereId)
            
            # Cleanup spheres in contact with robot
            robot_contacts = p.getContactPoints(sphereId, robotId)
            if robot_contacts and sphereId in self.sphereIds:
                p.removeBody(sphereId)
                self.sphereIds.remove(sphereId)

    def close(self):
        if p.isConnected():
            p.disconnect()


class OT2Env(gym.Env):
    """Gym environment for OT-2 robot with velocity-aware control"""
    metadata = {"render_modes": ["human", "rgb_array"], "render_fps": 240}

    def __init__(self, render=False):
        super().__init__()
        self.render_mode = "human" if render else "rgb_array"
        self.sim = Simulation(num_agents=1, render=render, rgb_array=(self.render_mode == "rgb_array"))
        
        # Action space: normalized velocities [-1, 1] for x,y,z axes
        self.action_space = spaces.Box(low=-1.0, high=1.0, shape=(3,), dtype=np.float32)
        
        # Enhanced observation space with velocity awareness
        self.observation_space = spaces.Box(
            low=-1.0, 
            high=1.0, 
            shape=(10,),  # [pos(3), goal(3), velocity(3), distance(1)]
            dtype=np.float32
        )
        
        # Workspace boundaries (in meters)
        self.workspace_low = np.array([-0.1871, -0.1706, 0.1195], dtype=np.float32)
        self.workspace_high = np.array([0.2532, 0.2197, 0.2897], dtype=np.float32)
        
        # Robot parameters
        self.base_position = np.array([0.0, 0.0, 0.03], dtype=np.float32)
        self.pipette_offset = np.array([0.073, 0.0895, 0.0895], dtype=np.float32)
        self.joint_limits = [(-0.25, 0.25), (-0.25, 0.25), (0.0, 0.17)]
        
        # State tracking
        self.prev_position = None
        self.current_velocity = np.zeros(3)
        self.max_steps = 1000
        self.steps = 0
        self.goal_pos = np.zeros(3)
        self.prev_dist = 0.0

    def _normalize_pos(self, pos):
        """Normalize position to [-1, 1] range based on workspace limits"""
        return 2.0 * (pos - self.workspace_low) / (self.workspace_high - self.workspace_low) - 1.0

    def _normalize_velocity(self, velocity):
        """Normalize velocity to [-1, 1] range (max expected velocity: 0.1 m/s)"""
        max_vel = 0.1
        return np.clip(velocity / max_vel, -1.0, 1.0)

    def _compute_velocity(self, current_pos, dt=1/240.0):
        """Compute velocity based on position change"""
        if self.prev_position is None:
            self.prev_position = current_pos.copy()
            return np.zeros(3)
        
        velocity = (current_pos - self.prev_position) / dt
        self.prev_position = current_pos.copy()
        return velocity

    def _validate_position(self, pos):
        """Ensure position stays within workspace boundaries"""
        return np.clip(pos, self.workspace_low, self.workspace_high)

    def _compute_joint_positions(self, target_pos):
        """Calculate joint positions to reach target pipette position"""
        return np.array([
            self.base_position[0] + self.pipette_offset[0] - target_pos[0],
            self.base_position[1] + self.pipette_offset[1] - target_pos[1],
            target_pos[2] - self.base_position[2] - self.pipette_offset[2]
        ], dtype=np.float32)

    def _validate_joint_limits(self, joints):
        """Enforce joint limits to prevent simulation instability"""
        return np.array([
            np.clip(joints[0], *self.joint_limits[0]),
            np.clip(joints[1], *self.joint_limits[1]),
            np.clip(joints[2], *self.joint_limits[2])
        ], dtype=np.float32)

    def _create_observation(self, current_pos):
        """Create observation with position, goal, velocity, and distance"""
        distance = np.linalg.norm(current_pos - self.goal_pos)
        max_workspace_dist = np.linalg.norm(self.workspace_high - self.workspace_low)
        distance_normalized = distance / max_workspace_dist
        
        return np.concatenate([
            self._normalize_pos(current_pos),
            self._normalize_pos(self.goal_pos),
            self._normalize_velocity(self.current_velocity),
            np.array([distance_normalized * 2 - 1])  # Scale to [-1, 1]
        ]).astype(np.float32)

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self.steps = 0
        self.prev_position = None
        self.current_velocity = np.zeros(3)
        
        # Randomize goal position within workspace
        self.goal_pos = self._validate_position(self.np_random.uniform(
            self.workspace_low, 
            self.workspace_high
        ))
        
        # Reset simulation
        state_dict = self.sim.reset(num_agents=1)
        robot_id = self.sim.robotIds[0]
        
        # Randomize starting position with validation
        for _ in range(10):
            start_pos = self._validate_position(self.np_random.uniform(
                self.workspace_low, 
                self.workspace_high
            ))
            joints = self._compute_joint_positions(start_pos)
            valid_joints = self._validate_joint_limits(joints)
            
            # Verify achievable position
            achieved_pos = np.array([
                self.base_position[0] + self.pipette_offset[0] - valid_joints[0],
                self.base_position[1] + self.pipette_offset[1] - valid_joints[1],
                self.base_position[2] + self.pipette_offset[2] + valid_joints[2]
            ])
            
            if np.all(achieved_pos >= self.workspace_low) and np.all(achieved_pos <= self.workspace_high):
                break
        
        # Apply joint positions
        for joint_idx, joint_val in enumerate(valid_joints):
            p.resetJointState(robot_id, joint_idx, targetValue=float(joint_val))
        
        # Get initial state
        state_dict = self.sim.get_states()
        current_pos = np.array(state_dict[f'robotId_{robot_id}']['pipette_position'], dtype=np.float32)
        self.prev_dist = np.linalg.norm(current_pos - self.goal_pos)
        self.prev_position = current_pos.copy()
        
        return self._create_observation(current_pos), {
            "start_pos": start_pos, 
            "goal_pos": self.goal_pos,
            "robot_id": robot_id
        }

    def step(self, action):
        self.steps += 1
        
        # Adaptive velocity control based on distance to goal
        state_dict = self.sim.get_states()
        robot_id = next(iter(state_dict))
        current_pos = np.array(state_dict[robot_id]['pipette_position'], dtype=np.float32)
        distance_to_goal = np.linalg.norm(current_pos - self.goal_pos)
        
        # Scale max velocity based on proximity to target
        max_velocity = 0.05 * min(1.0, max(0.1, distance_to_goal / 0.05))
        velocity = np.clip(action, -1.0, 1.0) * max_velocity
        
        # Execute action
        self.sim.run([[float(velocity[0]), float(velocity[1]), float(velocity[2]), 0.0]])
        
        # Update state
        state_dict = self.sim.get_states()
        current_pos = np.array(state_dict[robot_id]['pipette_position'], dtype=np.float32)
        self.current_velocity = self._compute_velocity(current_pos)
        curr_dist = np.linalg.norm(current_pos - self.goal_pos)
        
        # Reward calculation
        progress = self.prev_dist - curr_dist
        reward = progress * 2000.0  # High weight for progress
        
        # Velocity penalties and bonuses
        speed = np.linalg.norm(self.current_velocity)
        
        # Strong penalty for high speed when close to target
        if curr_dist < 0.01:  # Within 1cm
            velocity_penalty = (speed / 0.01) ** 2 * 4.0
            reward -= velocity_penalty
        
        # Smooth approach bonus
        if curr_dist < 0.02 and speed < 0.005:  # Within 2cm and slow
            reward += 1.0 - (speed / 0.005)
        
        # Terminal conditions
        terminated = False
        if curr_dist < 0.001:  # 1mm precision
            stopping_bonus = max(0.0, 10.0 - speed * 1000)  # Bonus for stopping
            reward += 100.0 + stopping_bonus
            if self.steps > self.max_steps * 0.7:
                reward += 20.0  # Stability bonus
            terminated = True
        
        # Instability penalty
        if np.any(np.abs(self.current_velocity) > 0.5):
            reward -= 50.0
            terminated = True
        
        # Time penalty
        reward -= 0.01
        
        # Update tracking
        self.prev_dist = curr_dist
        truncated = self.steps >= self.max_steps
        
        # Return step results
        return (
            self._create_observation(current_pos),
            float(reward),
            terminated,
            truncated,
            {
                'distance': curr_dist,
                'velocity': speed,
                'position': current_pos.copy(),
                'progress': progress
            }
        )

    def render(self):
        if self.render_mode == "rgb_array" and hasattr(self.sim, 'current_frame'):
            return self.sim.current_frame
        return None

    def close(self):
        self.sim.close()