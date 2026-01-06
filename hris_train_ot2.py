import numpy as np
import os
import matplotlib
matplotlib.use('Agg') 
import matplotlib.pyplot as plt
import gc
from clearml import Task, Logger
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.vec_env import DummyVecEnv
from hris_ot2_gym_wrapper import OT2Env

class AdvancedMonitorCallback(BaseCallback):
    def __init__(self, check_freq=1000, checkpoint_freq=100000, save_path='./models', verbose=1):
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
        if self.n_calls % self.checkpoint_freq == 0:
            chk_name = f"checkpoint_{self.n_calls}"
            path = os.path.join(self.save_path, f"{chk_name}.zip")
            self.model.save(path)
            if Task.current_task():
                Task.current_task().upload_artifact(name=chk_name, artifact_object=path)

        info = self.locals['infos'][0]
        if self.locals['dones'][0]:
            dist_m = info.get('distance', 1.0)
            dist_mm = dist_m * 1000
            self.precision_buffer.append(dist_mm)
            self.success_buffer.append(1 if dist_mm <= 1.0 else 0)
            
            if len(self.precision_buffer) > 100:
                self.precision_buffer.pop(0)
                self.success_buffer.pop(0)

            if dist_m < self.best_precision:
                self.best_precision = dist_m
                if dist_m < 0.05: 
                    self._save_best_trajectory()

        if 'position' in info:
            self.current_trajectory.append(info['position'].copy())

        if self.n_calls % self.check_freq == 0:
            self._evaluate_and_log()
            gc.collect() 
            
        return True

    def _evaluate_and_log(self):
        avg_precision = np.mean(self.precision_buffer) if self.precision_buffer else 0
        success_rate = np.mean(self.success_buffer) * 100 if self.success_buffer else 0
        
        if self.clearml_logger:
            self.clearml_logger.report_scalar("Precision Tracker", "Rolling Avg Distance (mm)", avg_precision, self.n_calls)
            self.clearml_logger.report_scalar("Precision Tracker", "Success Rate (%)", success_rate, self.n_calls)
            self.clearml_logger.report_scalar("Precision Tracker", "All-Time Best (mm)", self.best_precision * 1000, self.n_calls)

        print(f"Step {self.n_calls:,} | Success: {success_rate:.1f}% | Avg Dist: {avg_precision:.2f}mm | Best: {self.best_precision*1000:.2f}mm")

    def _save_best_trajectory(self):
        if len(self.current_trajectory) > 0:
            try:
                traj = np.array(self.current_trajectory)
                plt.close('all') 
                
                fig = plt.figure(figsize=(10, 8))
                ax = fig.add_subplot(111, projection='3d')
                ax.plot(traj[:, 0], traj[:, 1], traj[:, 2], 'b-', alpha=0.6)
                ax.scatter(traj[-1, 0], traj[-1, 1], traj[-1, 2], c='r', s=100)
                ax.scatter(traj[0, 0], traj[0, 1], traj[0, 2], c='g', s=100)
                
                img_path = os.path.join(self.save_path, f'best_traj_{self.n_calls}.png')
                plt.savefig(img_path)
                plt.close(fig)
                
                if self.clearml_logger:
                    self.clearml_logger.report_image("Best Trajectories", "3D Plot", self.n_calls, img_path)
            except Exception:
                pass
            self.current_trajectory = [] 

def main():
    task = Task.init(
        project_name='Mentor Group - Myrthe/Group 1', 
        task_name='hris_Precision_Training_Final_v7', 
        task_type=Task.TaskTypes.training,
        reuse_last_task_id=False
    )
    
    task.set_repo(repo='https://github.com/AndriiRak243703/Y2B25_Task_11.git', branch='hris/rl-training')
    task.set_base_docker('deanis/2023y2b-rl:latest')
    task.set_packages(['tensorboard', 'clearml', 'gymnasium', 'stable-baselines3==2.2.1', 'pybullet==3.2.5', 'matplotlib'])
    task.execute_remotely(queue_name='default', exit_process=True)
    
    env = OT2Env(render=False)
    env = DummyVecEnv([lambda: env])
    
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
        ent_coef=0.001,
        vf_coef=0.6,
        max_grad_norm=0.5,
        normalize_advantage=True,
        verbose=1,
        tensorboard_log="./ppo_logs/",
        device="auto"
    )
    
    adv_callback = AdvancedMonitorCallback(check_freq=5000, save_path='./models', verbose=1)
    
    try:
        model.learn(total_timesteps=5_000_000, callback=adv_callback, tb_log_name="PPO_Precision", progress_bar=True)
        model.save("final_model")
        task.upload_artifact("final_model", "final_model.zip")
    finally:
        env.close()

if __name__ == "__main__":
    main()