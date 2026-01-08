import numpy as np
import os
import gc
from clearml import Task
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import BaseCallback, CheckpointCallback, CallbackList
from stable_baselines3.common.vec_env import DummyVecEnv
from hris_ot2_gym_wrapper import OT2Env
import subprocess
import sys

# Force installation at runtime if it's missing
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
        # 1. Record distance when episode ends (Fast)
        if self.locals['dones'][0]:
            dist_mm = self.locals['infos'][0].get('distance', 1.0) * 1000
            self.precision_buffer.append(dist_mm)
            if len(self.precision_buffer) > 50:
                self.precision_buffer.pop(0)

        # 2. Heavy Logic (Only runs once every 5000 steps)
        if self.n_calls % self.check_freq == 0 and self.precision_buffer:
            avg_dist = np.mean(self.precision_buffer)
            
            # Log to ClearML Chart
            self.logger.record("trajectory/avg_distance_mm", avg_dist)
            
            # Dynamic LR Adjustment
            if avg_dist < 1.0:
                new_lr = 5e-6
            elif avg_dist < 2.0:
                new_lr = 1e-5
            elif avg_dist < 5.0:
                new_lr = 2e-5
            elif avg_dist < 15.0:
                new_lr = 5e-5
            elif avg_dist < 45.0: 
                new_lr = 1e-4
            else:
                new_lr = 2.5e-4

            if new_lr != self.current_lr:
                self.current_lr = new_lr
                for param_group in self.model.policy.optimizer.param_groups:
                    param_group['lr'] = new_lr
                print(f"Step {self.n_calls} | Avg Dist: {avg_dist:.2f}mm | LR → {new_lr:.1e}")

            self.logger.record("train/learning_rate_dynamic", new_lr)
            
            # Garbage collect ONLY here
            gc.collect()
            
        return True

def main():
    # Initialize ClearML Task
    task = Task.init(project_name='Mentor Group - Myrthe/Group 1', task_name='hris_Precision_Stabilized_v1')
    task.execute_remotely(queue_name='default', exit_process=True)
    
    env = DummyVecEnv([lambda: OT2Env(render=False)])
    
    # Checkpoint Callback: Saves every 200k steps
    checkpoint_callback = CheckpointCallback(
        save_freq=200000, 
        save_path='./checkpoints/',
        name_prefix='ot2_model_checkpoint'
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
    
    # Combined callbacks
    callback_list = CallbackList([PrecisionLRScheduler(), checkpoint_callback])
    
    # Train for 5 million steps
    try:
        model.learn(total_timesteps=10_000_000, callback=callback_list)
        model.save("final_model")
        task.upload_artifact("final_model", "final_model.zip")
    except Exception as e:
        # Emergency save if it crashes
        model.save("crash_recovery_model")
        task.upload_artifact("crash_model", "crash_recovery_model.zip")
        print(f"Task failed but model saved: {e}")

if __name__ == "__main__":
    main()