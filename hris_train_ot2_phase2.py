import numpy as np
import os
import gc
from clearml import Task
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import BaseCallback, CheckpointCallback, CallbackList
from stable_baselines3.common.vec_env import DummyVecEnv
# Import the NEW phase 2 wrapper
from hris_ot2_gym_wrapper_phase2 import OT2Env
import subprocess
import sys

# Ensure Tensorboard
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

        # Periodic logic
        if self.n_calls % self.check_freq == 0 and self.precision_buffer:
            avg_dist = np.mean(self.precision_buffer)
            self.logger.record("trajectory/avg_distance_mm", avg_dist)
            
            # --- AGGRESSIVE LOCK-IN CURRICULUM ---
            # Faster drops to tighten the net immediately since we are already fine-tuning
            if avg_dist < 1.0: target_lr = 1e-6    # 1mm: Deep Surgical
            elif avg_dist < 5.0: target_lr = 5e-6  # 5mm: Fine Tuning
            elif avg_dist < 10.0: target_lr = 1e-5 # 10mm: Tighten Grip
            elif avg_dist < 20.0: target_lr = 2e-5 # 20mm: Lock-in
            elif avg_dist < 40.0: target_lr = 5e-5 # 40mm: Approach
            else: target_lr = 1e-4

            # MONOTONIC LOCK
            if target_lr < self.current_lr:
                self.current_lr = target_lr
                for param_group in self.model.policy.optimizer.param_groups:
                    param_group['lr'] = self.current_lr
                print(f"MILESTONE REACHED: Step {self.n_calls} | Dist: {avg_dist:.2f}mm | LR Locked at {self.current_lr:.1e}")
            
            self.logger.record("train/learning_rate_dynamic", self.current_lr)
            gc.collect() 
        return True

def main():
    # 1. ClearML Init
    task = Task.init(
        project_name='Mentor Group - Myrthe/Group 1', 
        task_name='hris_Precision_Phase2_LockIn',
        output_uri=True 
    )
    task.execute_remotely(queue_name='default', exit_process=True)
    
    env = DummyVecEnv([lambda: OT2Env(render=False)])
    
    # 2. LOAD EXISTING MODEL
    model_path = "final_model.zip"
    if not os.path.exists(model_path):
        print("ERROR: final_model.zip not found! Cannot start fine-tuning.")
        return

    print(f"Loading model from {model_path} for Phase 2 Lock-In...")
    model = PPO.load(model_path, env=env)

    # 3. APPLY LOCK-IN OVERRIDES
    # Drastic reduction in randomness and update size
    model.ent_coef = 0.00005     # Virtually zero randomness
    model.clip_range = 0.1       # Restrict update size to 10% (prevents shock)
    model.learning_rate = 5e-5   # Start with a conservative base LR
    
    print("Lock-in Settings Applied: Ent=0.00005, Clip=0.1, LR=5e-5")

    # 4. CALLBACKS
    checkpoint_callback = CheckpointCallback(
        save_freq=200000, 
        save_path='./checkpoints/',
        name_prefix='ot2_phase2_lockin'
    )
    
    callback_list = CallbackList([PrecisionLRScheduler(check_freq=5000), checkpoint_callback])
    
    # 5. TRAIN
    try:
        # Run for 3 million steps to settle the model into the 1mm target
        model.learn(total_timesteps=5_000_000, callback=callback_list)
        model.save("phase2_final_lockin")
        task.upload_artifact("phase2_final_lockin", "phase2_final_lockin.zip")
        print("Phase 2 Lock-In Training Completed Successfully.")
    except Exception as e:
        print(f"Phase 2 Failed: {e}")
        model.save("phase2_crash_lockin")
        task.upload_artifact("phase2_crash_lockin", "phase2_crash_lockin.zip")

if __name__ == "__main__":
    main()