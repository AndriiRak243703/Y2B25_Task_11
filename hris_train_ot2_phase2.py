import numpy as np
import os
import gc
from clearml import Task
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import BaseCallback, CheckpointCallback, CallbackList
from stable_baselines3.common.vec_env import DummyVecEnv
# Import the NEW wrapper
from hris_ot2_gym_wrapper_phase2 import OT2Env
import subprocess
import sys

# Force installation of Tensorboard if missing
try:
    import tensorboard
except ImportError:
    subprocess.check_call([sys.executable, "-m", "pip", "install", "tensorboard"])

class PrecisionLRScheduler(BaseCallback):
    def __init__(self, check_freq=5000):
        super().__init__()
        self.check_freq = check_freq
        self.precision_buffer = []
        self.current_lr = 2.5e-4  # High Water Mark

    def _on_step(self) -> bool:
        if self.locals['dones'][0]:
            dist_mm = self.locals['infos'][0].get('distance', 1.0) * 1000
            self.precision_buffer.append(dist_mm)
            if len(self.precision_buffer) > 50:
                self.precision_buffer.pop(0)

        # Logic runs periodically to save CPU
        if self.n_calls % self.check_freq == 0 and self.precision_buffer:
            avg_dist = np.mean(self.precision_buffer)
            self.logger.record("trajectory/avg_distance_mm", avg_dist)
            
            # --- PHASE 2 CURRICULUM ---
            # Stricter, granular milestones to guide the model to 1mm
            if avg_dist < 1.0: target_lr = 5e-6
            elif avg_dist < 2.0: target_lr = 1e-5
            elif avg_dist < 5.0: target_lr = 2e-5
            elif avg_dist < 15.0: target_lr = 5e-5
            elif avg_dist < 30.0: target_lr = 7.5e-5
            elif avg_dist < 45.0: target_lr = 1e-4
            elif avg_dist < 60.0: target_lr = 1.75e-4
            else: target_lr = 2.5e-4

            # MONOTONIC LOCK: LR can only go down, never up
            if target_lr < self.current_lr:
                self.current_lr = target_lr
                for param_group in self.model.policy.optimizer.param_groups:
                    param_group['lr'] = self.current_lr
                print(f"MILESTONE REACHED: Step {self.n_calls} | Dist: {avg_dist:.2f}mm | LR Locked at {self.current_lr:.1e}")
            
            self.logger.record("train/learning_rate_dynamic", self.current_lr)
            gc.collect() 
        return True

def main():
    # 1. SETUP CLEARML with OUTPUT_URI to ensure files are uploaded
    task = Task.init(
        project_name='Mentor Group - Myrthe/Group 1', 
        task_name='hris_Precision_Phase2_FineTuning',
        output_uri=True 
    )
    task.execute_remotely(queue_name='default', exit_process=True)
    
    env = DummyVecEnv([lambda: OT2Env(render=False)])
    
    # 2. CHECK FOR PREVIOUS MODEL
    # Ensure 'final_model.zip' is in the root directory (uploaded via git or artifacts)
    model_path = "final_model.zip"
    if not os.path.exists(model_path):
        print("ERROR: final_model.zip not found! Cannot start fine-tuning.")
        return

    print(f"Loading model from {model_path} for Phase 2 Fine-Tuning...")
    model = PPO.load(model_path, env=env)

    # 3. APPLY FINE-TUNING OVERRIDES
    # Drastically reduce entropy (randomness) to stop the 'shaking'
    model.ent_coef = 0.0001
    print("Entropy Coefficient reduced to 0.0001 for precision.")

    # 4. CALLBACKS
    checkpoint_callback = CheckpointCallback(
        save_freq=200000, 
        save_path='./checkpoints/',
        name_prefix='ot2_phase2_model'
    )
    
    callback_list = CallbackList([PrecisionLRScheduler(check_freq=5000), checkpoint_callback])
    
    # 5. TRAIN
    try:
        # Run for 3 million steps to settle the model into the 1mm target
        model.learn(total_timesteps=5_000_000, callback=callback_list)
        model.save("phase2_final_model")
        # Explicit upload just in case
        task.upload_artifact("phase2_final_model", "phase2_final_model.zip")
        print("Phase 2 Training Completed Successfully.")
    except Exception as e:
        print(f"Phase 2 Failed: {e}")
        model.save("phase2_crash_model")
        task.upload_artifact("phase2_crash_model", "phase2_crash_model.zip")

if __name__ == "__main__":
    main()