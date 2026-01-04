import numpy as np
from clearml import Task, Logger
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.vec_env import DummyVecEnv
from ot2_gym_wrapper import OT2Env
import os
import time

class ClearMLCallback(BaseCallback):
    """Custom callback for logging metrics to ClearML"""
    def __init__(self, verbose=0):
        super().__init__(verbose)
        self.episode_rewards = []
        self.episode_lengths = []
        self.best_mean_reward = -np.inf
        self.last_log_step = 0
        self.log_freq = 1000  # Log every 1000 steps

    def _on_step(self) -> bool:
        if self.n_calls % self.log_freq == 0:
            # Log training metrics
            self.logger.record('time/total_timesteps', self.num_timesteps)
            self.logger.record('train/learning_rate', self.model.policy.optimizer.param_groups[0]['lr'])
            
            # Log episode statistics if available
            if len(self.episode_rewards) > 0:
                mean_reward = np.mean(self.episode_rewards[-10:])
                mean_length = np.mean(self.episode_lengths[-10:])
                self.logger.record('rollout/ep_rew_mean', mean_reward)
                self.logger.record('rollout/ep_len_mean', mean_length)
                
                # Save best model
                if mean_reward > self.best_mean_reward:
                    self.best_mean_reward = mean_reward
                    self.model.save("best_model")
                    if self.verbose > 0:
                        print(f"New best mean reward: {mean_reward:.2f} - Model saved!")
            
            # Clear buffers
            self.episode_rewards = []
            self.episode_lengths = []
        
        return True

    def _on_rollout_end(self):
        """Log at the end of each rollout"""
        if hasattr(self.model, 'ep_info_buffer') and self.model.ep_info_buffer is not None:
            for info in self.model.ep_info_buffer:
                if 'episode' in info.keys():
                    self.episode_rewards.append(info['episode']['r'])
                    self.episode_lengths.append(info['episode']['l'])

class EpisodeInfoLogger(BaseCallback):
    """Logs detailed episode information to ClearML"""
    def __init__(self, verbose=0):
        super().__init__(verbose)
        self.episode_num = 0

    def _on_step(self) -> bool:
        if self.locals.get('dones', [False])[0]:
            self.episode_num += 1
            info = self.locals['infos'][0]
            
            # Extract metrics
            distance = info.get('distance', 0.0)
            velocity = info.get('velocity', 0.0)
            position = info.get('position', np.zeros(3))
            
            # Log to ClearML
            logger = Logger.current_logger()
            logger.report_scalar(
                title="Precision Metrics",
                series="Final Distance (mm)",
                value=distance * 1000,
                iteration=self.episode_num
            )
            logger.report_scalar(
                title="Velocity Control",
                series="Final Speed (mm/s)",
                value=velocity * 1000,
                iteration=self.episode_num
            )
            logger.report_vector(
                title="Final Position",
                series="XYZ Coordinates",
                values=position.tolist(),
                iteration=self.episode_num
            )
            
            # Log trajectory if available
            if 'trajectory' in info:
                logger.report_line_plot(
                    title="Trajectory",
                    series="Path",
                    iteration=self.episode_num,
                    x_axis="X",
                    y_axis="Y",
                    x_values=[p[0] for p in info['trajectory']],
                    y_values=[p[1] for p in info['trajectory']]
                )
        return True

def main():
    # Initialize ClearML task
    task = Task.init(
        project_name='Mentor Group - Myrthe/Group 1', 
        task_name='hris_Precision_Training_v2', 
        task_type=Task.TaskTypes.training,
        reuse_last_task_id=False
    )
    
    # Configure environment
    task.set_repo(
        repo='https://github.com/AndriiRak243703/Y2B25_Task_11.git',
        branch='hris/rl-training'
    )
    task.set_base_docker('deanis/2023y2b-rl:latest')
    task.set_packages([
        'tensorboard', 'clearml', 'gymnasium', 
        'stable-baselines3==2.2.1', 'pybullet==3.2.5'
    ])
    task.execute_remotely(queue_name='default', exit_process=True)
    
    # Create environment
    env = OT2Env(render=False)
    env = DummyVecEnv([lambda: env])
    
    # Configure PPO with velocity-aware hyperparameters
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
    
    # Setup callbacks
    clearml_callback = ClearMLCallback(verbose=1)
    episode_logger = EpisodeInfoLogger(verbose=0)
    
    # Training loop
    print("--- STARTING PRECISION TRAINING ---")
    print(f"Task ID: {task.id}")
    print(f"Logging to: {task.get_output_log_web_page()}")
    
    try:
        model.learn(
            total_timesteps=5_000_000,
            callback=[clearml_callback, episode_logger],
            tb_log_name="PPO_VelocityControl",
            progress_bar=True
        )
        
        # Save final model
        model.save("final_model")
        task.upload_artifact("final_model", "final_model.zip")
        
        # Save best model
        if os.path.exists("best_model.zip"):
            task.upload_artifact("best_model", "best_model.zip")
        
        print("--- TRAINING COMPLETED SUCCESSFULLY ---")
    except Exception as e:
        print(f"Training failed: {str(e)}")
        task.get_logger().report_text(f"Training failed: {str(e)}")
        raise
    finally:
        env.close()

if __name__ == "__main__":
    main()