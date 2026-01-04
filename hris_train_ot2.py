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
    Enhanced monitoring to track how often the 1mm target is hit
    and save checkpoints every 100k steps.
    """
    def __init__(self, check_freq=1000, checkpoint_freq=100000, save_path='./models', verbose=1):
        super().__init__(verbose)
        self.check_freq = check_freq
        self.checkpoint_freq = checkpoint_freq
        self.save_path = save_path
        
        # Performance Tracking
        self.best_mean_reward = -np.inf
        self.best_precision = np.inf  
        self.last_improvement = 0
        self.max_no_improvement = 500_000  # Relaxed stopping
        
        # Precision Tracking Buffers
        self.precision_buffer = [] # Stores final distance of last N episodes
        self.success_buffer = []   # Stores 1 if <1mm, 0 otherwise
        
        self.current_trajectory = []
        os.makedirs(save_path, exist_ok=True)

    def _init_callback(self) -> None:
        self.clearml_logger = Logger.current_logger()

    def _on_step(self) -> bool:
        # Checkpoint logic
        if self.n_calls % self.checkpoint_freq == 0:
            chk_name = f"checkpoint_{self.n_calls}"
            path = os.path.join(self.save_path, f"{chk_name}.zip")
            self.model.save(path)
            if Task.current_task():
                Task.current_task().upload_artifact(name=chk_name, artifact_object=path)
        
        # Tracking trajectory for plotting
        if 'position' in self.locals['infos'][0]:
            self.current_trajectory.append(self.locals['infos'][0]['position'].copy())

        # Logging logic
        if self.n_calls % self.check_freq == 0:
            self._evaluate_and_log()
            
        return True

    def _on_rollout_end(self) -> None:
        if self.locals['dones'][0]: 
            info = self.locals['infos'][0]
            dist_m = info.get('distance', 1.0)
            dist_mm = dist_m * 1000
            
            # Update Buffers
            self.precision_buffer.append(dist_mm)
            self.success_buffer.append(1 if dist_mm <= 1.0 else 0)
            
            # Keep buffers at a reasonable size (last 100 episodes)
            if len(self.precision_buffer) > 100:
                self.precision_buffer.pop(0)
                self.success_buffer.pop(0)
            
            # Log current best
            if dist_m < self.best_precision:
                self.best_precision = dist_m
                self.last_improvement = self.n_calls
                if dist_m < 0.001:
                    self._save_best_trajectory()
            
            self.current_trajectory = []

    def _evaluate_and_log(self):
        # Calculate Rolling Metrics
        avg_precision = np.mean(self.precision_buffer) if self.precision_buffer else 0
        success_rate = np.mean(self.success_buffer) * 100 if self.success_buffer else 0
        
        if self.clearml_logger:
            # New specific graph for Precision Progress
            self.clearml_logger.report_scalar(
                "Precision Tracker", "Rolling Avg Distance (mm)", avg_precision, self.n_calls)
            self.clearml_logger.report_scalar(
                "Precision Tracker", "Success Rate (%)", success_rate, self.n_calls)
            self.clearml_logger.report_scalar(
                "Precision Tracker", "All-Time Best (mm)", self.best_precision * 1000, self.n_calls)

        # Print detailed status
        print(f"Step {self.n_calls:,} | Success: {success_rate:.1f}% | Avg Dist: {avg_precision:.2f}mm | Best: {self.best_precision*1000:.2f}mm")

    def _save_best_trajectory(self):
        # [Existing trajectory plotting code here...]
        pass

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