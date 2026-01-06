import numpy as np
import os
import gc
from clearml import Task
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.vec_env import DummyVecEnv
from hris_ot2_gym_wrapper import OT2Env

class PrecisionLRScheduler(BaseCallback):
    def __init__(self, check_freq=5000):
        super().__init__()
        self.check_freq = check_freq
        self.precision_buffer = []
        self.current_lr = 2.5e-4

    def _on_step(self) -> bool:
        # Log distance only on episode done
        if self.locals['dones'][0]:
            dist_mm = self.locals['infos'][0].get('distance', 1.0) * 1000
            self.precision_buffer.append(dist_mm)
            if len(self.precision_buffer) > 50:
                self.precision_buffer.pop(0)

        # Adjust LR periodically
        if self.n_calls % self.check_freq == 0 and self.precision_buffer:
            avg_dist = np.mean(self.precision_buffer)
            # Dynamic LR based on precision
            if avg_dist < 1.0:
                new_lr = 5e-6    # Very fine tuning
            elif avg_dist < 2.0:
                new_lr = 1e-5
            elif avg_dist < 5.0:
                new_lr = 5e-5
            elif avg_dist < 15.0:
                new_lr = 1e-4
            else:
                new_lr = 2.5e-4  # Default

            if new_lr != self.current_lr:
                self.current_lr = new_lr
                # Update optimizer LR
                for param_group in self.model.policy.optimizer.param_groups:
                    param_group['lr'] = new_lr
                print(f"Step {self.n_calls} | Avg Dist: {avg_dist:.2f}mm | LR → {new_lr:.1e}")
            else:
                print(f"Step {self.n_calls} | Avg Dist: {avg_dist:.2f}mm | LR: {self.current_lr:.1e}")
            gc.collect()
        return True

def main():
    task = Task.init(project_name='Mentor Group - Myrthe/Group 1', task_name='hris_Precision_Final_DynamicLR')
    task.execute_remotely(queue_name='default', exit_process=True)
    
    env = DummyVecEnv([lambda: OT2Env(render=False)])
    
    model = PPO(
        "MlpPolicy",
        env,
        learning_rate=2.5e-4,
        n_steps=2048,
        batch_size=64,
        gamma=0.995,        # Slightly reduced
        ent_coef=0.001,
        verbose=1,
        tensorboard_log="./ppo_ot2_tensorboard/"
    )
    
    # Train for 5 million steps
    model.learn(total_timesteps=5_000_000, callback=PrecisionLRScheduler())
    model.save("final_model")
    task.upload_artifact("final_model", "final_model.zip")

if __name__ == "__main__":
    main()