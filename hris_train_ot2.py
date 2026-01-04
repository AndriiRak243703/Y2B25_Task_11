import numpy as np
import os
import matplotlib
# Set non-interactive backend for headless docker environments
matplotlib.use('Agg') 
import matplotlib.pyplot as plt
from clearml import Task, Logger
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.vec_env import DummyVecEnv
from hris_ot2_gym_wrapper import OT2Env

class AdvancedMonitorCallback(BaseCallback):
    """
    Comprehensive monitoring with visualization, early stopping, 
    and ClearML integration.
    """
    def __init__(self, check_freq=1000, save_path='./models', verbose=1):
        super().__init__(verbose)
        self.check_freq = check_freq
        self.save_path = save_path
        self.best_mean_reward = -np.inf
        self.best_precision = np.inf  # Best distance to target
        self.last_improvement = 0
        self.max_no_improvement = 50_000  # Steps without improvement
        
        # Data buffers
        self.current_trajectory = []
        self.velocity_samples = []  # Added buffer for velocity
        self.best_trajectories = []
        
        # Ensure save directory exists
        os.makedirs(save_path, exist_ok=True)

    def _init_callback(self) -> None:
        # ClearML logger initialization
        self.clearml_logger = Logger.current_logger()
        
        # Register custom plots layout in ClearML
        if self.clearml_logger:
            self.clearml_logger.report_scatter2d(
                title="Workspace Coverage",
                series="Positions",
                iteration=0,
                xaxis="X Position",
                yaxis="Y Position",
                mode='markers'
            )

    def _on_step(self) -> bool:
        # 1. Capture Info from the environment
        info = self.locals['infos'][0] # Assuming Single Env (DummyVecEnv)
        
        # 2. Trajectory Tracking
        if 'position' in info:
            pos = info['position']
            self.current_trajectory.append(pos.copy())
            
        # 3. Velocity Tracking (Fixed: Populate the buffer)
        if 'velocity' in info:
            self.velocity_samples.append(info['velocity'])

        # 4. Periodic evaluation
        if self.n_calls % self.check_freq == 0:
            self._evaluate_and_log()
        
        # 5. Early stopping check
        if self.n_calls - self.last_improvement > self.max_no_improvement:
            if self.verbose > 0:
                print(f"🛑 Stopping early after {self.n_calls} steps with no improvement")
            return False  # Stop training
        
        return True

    def _on_rollout_end(self) -> None:
        """End of rollout handling (updates graphs and scalar metrics)"""
        # We check if the environment is actually done
        if self.locals['dones'][0]: 
            info = self.locals['infos'][0]
            
            # Log precision metrics
            distance = info.get('distance', 0.0)
            velocity = info.get('velocity', 0.0)
            
            if self.clearml_logger:
                self.clearml_logger.report_scalar(
                    "Precision Metrics", "Final Distance (mm)", distance * 1000, self.n_calls)
                self.clearml_logger.report_scalar(
                    "Velocity Control", "Final Speed (mm/s)", velocity * 1000, self.n_calls)
            
            # Track best precision
            if distance < self.best_precision:
                self.best_precision = distance
                self.last_improvement = self.n_calls
                # Save trajectory if sub-millimeter precision
                if distance < 0.001: 
                    self._save_best_trajectory()
            
            # Reset trajectory buffer for the next episode
            self.current_trajectory = []

    def _evaluate_and_log(self):
        """Comprehensive evaluation and logging"""
        # Get current policy performance
        episode_rewards = self.model.ep_info_buffer or []
        if episode_rewards and len(episode_rewards) > 0:
            mean_reward = np.mean([ep['r'] for ep in episode_rewards])
            mean_length = np.mean([ep['l'] for ep in episode_rewards])
            
            # Update best model if improved based on Reward
            if mean_reward > self.best_mean_reward:
                self.best_mean_reward = mean_reward
                self.model.save(os.path.join(self.save_path, "best_model"))
                
                # Upload artifact
                if Task.current_task():
                    Task.current_task().upload_artifact(
                        name=f"best_model_step_{self.n_calls}", 
                        artifact_object=os.path.join(self.save_path, "best_model.zip")
                    )
            
            # Log general training metrics
            if self.clearml_logger:
                self.clearml_logger.report_scalar(
                    "Training", "Mean Reward", mean_reward, self.n_calls)
                self.clearml_logger.report_scalar(
                    "Training", "Episode Length", mean_length, self.n_calls)
            
            # Learning rate monitoring
            lr = self.model.policy.optimizer.param_groups[0]['lr']
            if self.clearml_logger:
                self.clearml_logger.report_scalar("Training", "Learning Rate", lr, self.n_calls)
            
            # Velocity distribution analysis
            if len(self.velocity_samples) > 0:
                avg_speed = np.mean(self.velocity_samples)
                max_speed = np.max(self.velocity_samples)
                if self.clearml_logger:
                    self.clearml_logger.report_scalar(
                        "Velocity Analysis", "Average Speed (m/s)", avg_speed, self.n_calls)
                    self.clearml_logger.report_scalar(
                        "Velocity Analysis", "Max Speed (m/s)", max_speed, self.n_calls)
                # Clear buffer
                self.velocity_samples = []
            
            # Print console status
            print(f"Step {self.n_calls:,} | "
                  f"Mean Reward: {mean_reward:.2f} | "
                  f"Best Precision: {self.best_precision*1000:.2f}mm | "
                  f"LR: {lr:.2e}")

    def _save_best_trajectory(self):
        """Save and visualize best trajectories"""
        if len(self.current_trajectory) > 0:
            self.best_trajectories.append(self.current_trajectory.copy())
            
            # Create trajectory visualization
            try:
                fig = plt.figure(figsize=(10, 8))
                ax = fig.add_subplot(111, projection='3d')
                
                traj = np.array(self.current_trajectory)
                # Plot path
                ax.plot(traj[:, 0], traj[:, 1], traj[:, 2], 'b-', linewidth=2, label='Path')
                # Plot End Point
                ax.scatter(traj[-1, 0], traj[-1, 1], traj[-1, 2], c='r', s=100, label='Final Position')
                # Plot Start Point
                ax.scatter(traj[0, 0], traj[0, 1], traj[0, 2], c='g', s=50, label='Start Position')
                
                ax.set_xlabel('X (m)')
                ax.set_ylabel('Y (m)')
                ax.set_zlabel('Z (m)')
                ax.set_title(f'Best Trajectory (Final Dist: {self.best_precision*1000:.2f}mm)')
                ax.legend()
                
                # Save and log to ClearML
                img_path = os.path.join(self.save_path, f'trajectory_{self.n_calls}.png')
                plt.savefig(img_path)
                plt.close(fig)
                
                if self.clearml_logger:
                    self.clearml_logger.report_image(
                        title="Best Trajectories", 
                        series="3D Plot", 
                        iteration=self.n_calls,
                        local_path=img_path
                    )
            except Exception as e:
                print(f"Error plotting trajectory: {e}")

def main():
    # 1. Initialize ClearML task
    task = Task.init(
        project_name='Mentor Group - Myrthe/Group 1', 
        task_name='hris_Precision_Training_Adv_Monitor', 
        task_type=Task.TaskTypes.training,
        reuse_last_task_id=False
    )
    
    # 2. Configure environment settings
    task.set_repo(
        repo='https://github.com/AndriiRak243703/Y2B25_Task_11.git',
        branch='hris/rl-training'
    )
    task.set_base_docker('deanis/2023y2b-rl:latest')
    
    # Added matplotlib to packages for the Advanced Callback
    task.set_packages([
        'tensorboard', 
        'clearml', 
        'gymnasium', 
        'stable-baselines3==2.2.1', 
        'pybullet==3.2.5',
        'matplotlib' 
    ])
    
    task.execute_remotely(queue_name='default', exit_process=True)
    
    # 3. Create environment
    env = OT2Env(render=False)
    env = DummyVecEnv([lambda: env])
    
    # 4. Configure PPO with velocity-aware hyperparameters
    model = PPO(
        "MlpPolicy",
        env,
        learning_rate=2.5e-4,
        n_steps=2048,
        batch_size=64,
        n_epochs=10,
        gamma=0.998,
        gae_lambda=0.98,
        clip_range=0.15,
        ent_coef=0.0005,
        vf_coef=0.6,
        max_grad_norm=0.3,
        normalize_advantage=True,
        verbose=1,
        tensorboard_log="./ppo_logs/",
        device="auto"
    )
    
    # 5. Initialize the Advanced Callback
    adv_callback = AdvancedMonitorCallback(
        check_freq=1000, 
        save_path='./models', 
        verbose=1
    )
    
    # 6. Training loop
    print("--- STARTING PRECISION TRAINING (ADVANCED MONITORING) ---")
    print(f"Task ID: {task.id}")
    print(f"Logging to: {task.get_output_log_web_page()}")
    
    try:
        model.learn(
            total_timesteps=5_000_000,
            callback=adv_callback,
            tb_log_name="PPO_VelocityControl",
            progress_bar=True
        )
        
        # Save final model
        model.save("final_model")
        task.upload_artifact("final_model", "final_model.zip")
        
        print("--- TRAINING COMPLETED SUCCESSFULLY ---")
        
    except Exception as e:
        print(f"Training failed: {str(e)}")
        # Report failure to ClearML text log
        Logger.current_logger().report_text(f"Training failed: {str(e)}")
        raise
    finally:
        env.close()

if __name__ == "__main__":
    main()