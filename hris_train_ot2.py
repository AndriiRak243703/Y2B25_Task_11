import numpy as np
import os
import gc
from clearml import Task
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.vec_env import DummyVecEnv
from hris_ot2_gym_wrapper import OT2Env

class MonitorCallback(BaseCallback):
    def __init__(self, check_freq=5000, save_path='./models'):
        super().__init__()
        self.check_freq = check_freq
        self.save_path = save_path
        self.best_precision = np.inf
        os.makedirs(save_path, exist_ok=True)

    def _on_step(self) -> bool:
        if self.locals['dones'][0]:
            dist_mm = self.locals['infos'][0].get('distance', 1.0) * 1000
            if dist_mm < self.best_precision: self.best_precision = dist_mm
            
        if self.n_calls % self.check_freq == 0:
            print(f"Step {self.n_calls} | Best: {self.best_precision:.2f}mm")
            gc.collect() # Prevent memory leak
        return True

def main():
    task = Task.init(project_name='Mentor Group - Myrthe/Group 1', task_name='hris_Precision_Training_Final_v8')
    task.execute_remotely(queue_name='default', exit_process=True)
    
    env = DummyVecEnv([lambda: OT2Env(render=False)])
    model = PPO("MlpPolicy", env, learning_rate=2.5e-4, n_steps=2048, batch_size=64, 
                gamma=0.998, ent_coef=0.001, verbose=1)
    
    model.learn(total_timesteps=5_000_000, callback=MonitorCallback())
    model.save("final_model")
    task.upload_artifact("final_model", "final_model.zip")

if __name__ == "__main__":
    main()