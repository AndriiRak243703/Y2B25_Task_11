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
    Comprehensive monitoring with periodic check-pointing, 
    visualization, and relaxed early stopping.
    """
    def __init__(self, check_freq=1000, checkpoint_freq=100000, save_path='./models', verbose=1):
        super().__init__(verbose)
        self.check_freq = check_freq
        self.checkpoint_freq = checkpoint_freq
        self.save_path = save_path
        self.best_mean_reward = -np.inf
        self.best_precision = np.inf  
        self.last_improvement = 0
        # Increased to 500k to allow for long-term refinement
        self.max_no_improvement = 500_000  
        
        self.current_trajectory = []
        self.velocity_samples = []  
        os.makedirs(save_path, exist_ok=True)

    def _init_callback(self) -> None:
        self.clearml_logger = Logger.current_logger()

    def _on_step(self) -> bool:
        info = self.locals['infos'][0]
        
        # Track data for logging
        if 'position' in info:
            self.current_trajectory.append(info['position'].copy())
        if 'velocity' in info:
            self.velocity_samples.append(info['velocity'])

        # 1. Periodic Evaluation & Logging
        if self.n_calls % self.check_freq == 0:
            self._evaluate_and_log()
            
        # 2. Periodic Checkpointing (Every 100k steps)
        if self.n_calls % self.checkpoint_freq == 0:
            chk_name = f"checkpoint_{self.n_calls}"
            path = os.path.join(self.save_path, f"{chk_name}.zip")
            self.model.save(path)
            if Task.current_task():
                Task.current_task().upload_artifact(name=chk_name, artifact_object=path)
            if self.verbose > 0:
                print(f"💾 Saved checkpoint at step {self.n_calls}")
        
        # 3. Early stopping check (relaxed)
        if self.n_calls - self.last_improvement > self.max_no_improvement:
            if self.verbose > 0:
                print(f"🛑 Stopping early: No improvement in {self.max_no_improvement} steps")
            return False  
        
        return True

    def _on_rollout_end(self) -> None:
        if self.locals['dones'][0]: 
            info = self.locals['infos'][0]
            dist = info.get('distance', 1.0)
            
            # Log precision to ClearML
            if self.clearml_logger:
                self.clearml_logger.report_scalar("Precision", "Dist (mm)", dist * 1000, self.n_calls)
            
            # Record best precision ever seen
            if dist < self.best_precision:
                self.best_precision = dist
                self.last_improvement = self.n_calls # Reset improvement clock
                if dist < 0.001: # Sub-millimeter
                    self._save_best_trajectory()
            
            self.current_trajectory = []

    def _evaluate_and_log(self):
        episode_rewards = self.model.ep_info_buffer or []
        if episode_rewards and len(episode_rewards) > 0:
            mean_reward = np.mean([ep['r'] for ep in episode_rewards])
            
            # Update best model based on reward
            if mean_reward > self.best_mean_reward:
                self.best_mean_reward = mean_reward
                self.last_improvement = self.n_calls
                self.model.save(os.path.join(self.save_path, "best_model"))
            
            if self.clearml_logger:
                self.clearml_logger.report_scalar("Training", "Mean Reward", mean_reward, self.n_calls)
            
            # Print status to console
            print(f"Step {self.n_calls:,} | Reward: {mean_reward:.2f} | Best Precision: {self.best_precision*1000:.2f}mm")

    def _save_best_trajectory(self):
        if len(self.current_trajectory) > 0:
            try:
                traj = np.array(self.current_trajectory)
                fig = plt.figure(figsize=(10, 8))
                ax = fig.add_subplot(111, projection='3d')
                ax.plot(traj[:, 0], traj[:, 1], traj[:, 2], 'b-')
                ax.scatter(traj[-1, 0], traj[-1, 1], traj[-1, 2], c='r', s=100)
                img_path = os.path.join(self.save_path, f'best_traj_{self.n_calls}.png')
                plt.savefig(img_path)
                plt.close(fig)
                if self.clearml_logger:
                    self.clearml_logger.report_image("Best Trajectories", "3D Plot", self.n_calls, img_path)
            except Exception as e:
                print(f"Plotting failed: {e}")

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