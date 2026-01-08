import numpy as np
import os
import gc
from clearml import Task
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import BaseCallback, CheckpointCallback, CallbackList
from stable_baselines3.common.vec_env import DummyVecEnv
# Import the updated environment
from final_attempt_wrapper import OT2Env  # ← changed from hris_ot2_gym_wrapper
import subprocess
import sys

try:
    import tensorboard
except ImportError:
    subprocess.check_call([sys.executable, "-m", "pip", "install", "tensorboard"])


class PrecisionLRScheduler(BaseCallback):
    def __init__(self, check_freq=5000):
        super().__init__()
        self.check_freq = check_freq
        self.precision_buffer = []
        self.current_lr = 2.5e-4

    def _on_step(self) -> bool:
        if self.locals['dones'][0]:
            dist_mm = self.locals['infos'][0].get('distance', 1.0) * 1000
            self.precision_buffer.append(dist_mm)
            if len(self.precision_buffer) > 50:
                self.precision_buffer.pop(0)

        if self.n_calls % self.check_freq == 0 and self.precision_buffer:
            avg_dist = np.mean(self.precision_buffer)
            self.logger.record("trajectory/avg_distance_mm", avg_dist)
            
            if avg_dist < 1.0: 
                target_lr = 5e-6
            elif avg_dist < 2.0: 
                target_lr = 1e-5
            elif avg_dist < 5.0: 
                target_lr = 2e-5
            elif avg_dist < 15.0: 
                target_lr = 5e-5
            elif avg_dist < 30.0: 
                target_lr = 7.5e-5
            elif avg_dist < 45.0: 
                target_lr = 1e-4
            elif avg_dist < 60.0: 
                target_lr = 1.75e-4
            else: 
                target_lr = 2.5e-4

            if target_lr < self.current_lr:
                self.current_lr = target_lr
                for param_group in self.model.policy.optimizer.param_groups:
                    param_group['lr'] = self.current_lr
                print(f"MILESTONE REACHED: Step {self.n_calls} | Dist: {avg_dist:.2f}mm | LR Locked at {self.current_lr:.1e}")
            
            self.logger.record("train/learning_rate_dynamic", self.current_lr)
            gc.collect() 
            
        return True


def main():
    task = Task.init(project_name='Mentor Group - Myrthe/Group 1', task_name='hris_Precision_Final_Curriculum')
    task.execute_remotely(queue_name='default', exit_process=True)
    
    env = DummyVecEnv([lambda: OT2Env(render=False)])
    
    checkpoint_callback = CheckpointCallback(
        save_freq=200000, 
        save_path='./checkpoints/',
        name_prefix='ot2_curriculum_model'
    )

    model = PPO(
        "MlpPolicy",
        env,
        device="cpu",
        learning_rate=2.5e-4,
        n_steps=2048,
        batch_size=64,
        gamma=0.995,
        ent_coef=0.001,
        verbose=1,
        tensorboard_log="./ppo_ot2_tensorboard/"
    )
    
    callback_list = CallbackList([PrecisionLRScheduler(check_freq=5000), checkpoint_callback])
    
    try:
        print("Starting Curriculum Training...")
        model.learn(total_timesteps=10_000_000, callback=callback_list)
        model.save("final_model")
        task.upload_artifact("final_model", "final_model.zip")
    except Exception as e:
        print(f"Training interrupted: {e}")
        model.save("emergency_recovery_model")
        task.upload_artifact("crash_model", "emergency_recovery_model.zip")


if __name__ == "__main__":
    main()