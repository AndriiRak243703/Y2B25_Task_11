import numpy as np
import argparse
import plotly.graph_objects as go
from clearml import Task
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import BaseCallback
from hris_ot2_gym_wrapper import OT2Env

class MaximusCallback(BaseCallback):
    def __init__(self, log_freq=2048):
        super().__init__()
        self.log_freq = log_freq
        self.trajectory = []
        print(f"\n{'Step':<10} | {'Error (mm)':<12} | {'Status'}")
        print("-" * 45)

    def _on_step(self) -> bool:
        # Collect current 3D position (un-normalize for plotting)
        obs = self.locals['new_obs'][0]
        self.trajectory.append(obs[:3])

        if self.n_calls % self.log_freq == 0:
            # Calculate distance in mm for console
            dist_mm = (np.linalg.norm(obs[:3] - obs[3:]) * 100.0) # Approx scale
            status = "💎 PERFECT" if dist_mm < 1.0 else "🔥 HOT"
            print(f"{self.num_timesteps:<10} | {dist_mm:>10.2f} mm | {status}")
            
            # Send 3D Plot to ClearML
            self._report_plot()
        return True

    def _report_plot(self):
        path = np.array(self.trajectory)
        fig = go.Figure(data=[go.Scatter3d(
            x=path[:, 0], y=path[:, 1], z=path[:, 2],
            mode='lines',
            line=dict(color='red', width=3)
        )])
        fig.update_layout(title=f"Robot Trajectory at Step {self.num_timesteps}")
        
        Task.current_task().get_logger().report_plotly(
            title="Movement Path", series="Trajectory", iteration=self.num_timesteps, figure=fig
        )
        self.trajectory = [] # Reset for next batch

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--total_timesteps", type=int, default=2000000)
    args = parser.parse_args()

    # 1. ClearML Setup
    task = Task.init(
        project_name='Mentor Group - Jason/Group 1', 
        task_name='hris_Addiction_Maximus_FINAL_PLOT',
        reuse_last_task_id=False
    )
    
    task.set_repo(repo='https://github.com/AndriiRak243703/Y2B25_Task_11.git', branch='hris/rl-training')
    task.set_base_docker('deanis/2023y2b-rl:latest')
    task.set_packages(['tensorboard', 'clearml', 'gymnasium', 'stable-baselines3', 'pybullet', 'plotly'])

    task.execute_remotely(queue_name='default')

    # 2. Training
    env = OT2Env(render=False)
    model = PPO("MlpPolicy", env, learning_rate=5e-5, verbose=1, tensorboard_log="./logs/")

    print("--- DEPLOYING HRIS ADDICTION MAXIMUS WITH PLOTS ---")
    model.learn(total_timesteps=args.total_timesteps, callback=MaximusCallback())
    model.save("maximus_final_model")

if __name__ == "__main__":
    main()