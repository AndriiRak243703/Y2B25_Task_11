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
        # Collect 3D path for Plotly (normalized)
        obs = self.locals['new_obs'][0]
        self.trajectory.append(obs[:3])

        if self.n_calls % self.log_freq == 0:
            # Use the actual distance from the environment info if available
            dist_mm = np.linalg.norm(obs[:3] - obs[3:]) * 150 # Visual scale approx
            status = "💎 SETTLING" if dist_mm < 2.0 else "🔥 EXPLORING"
            print(f"{self.num_timesteps:<10} | {dist_mm:>10.2f} mm | {status}")
            self._report_plot()
        return True

    def _report_plot(self):
        path = np.array(self.trajectory)
        fig = go.Figure(data=[go.Scatter3d(
            x=path[:, 0], y=path[:, 1], z=path[:, 2],
            mode='lines', line=dict(color='lime', width=4)
        )])
        fig.update_layout(title=f"Path Trajectory - Step {self.num_timesteps}")
        Task.current_task().get_logger().report_plotly(
            title="3D Path", series="Trajectory", iteration=self.num_timesteps, figure=fig
        )
        self.trajectory = []

def main():
    task = Task.init(
        project_name='Mentor Group - Myrthe/Group 1', 
        task_name='hris_Addiction_Maximus_Precision_V1',
        reuse_last_task_id=False
    )
    
    task.set_repo(repo='https://github.com/AndriiRak243703/Y2B25_Task_11.git', branch='hris/rl-training')
    task.set_base_docker('deanis/2023y2b-rl:latest')
    task.set_packages(['tensorboard', 'clearml', 'gymnasium', 'stable-baselines3', 'pybullet', 'plotly'])
    task.execute_remotely(queue_name='default')

    env = OT2Env(render=False)
    
    # Brain Hyperparameters tuned to break the 80mm loop
    model = PPO(
        "MlpPolicy", 
        env, 
        learning_rate=3e-5,    # Slower, more precise learning
        gamma=0.99,            # Focus on long-term reward (the 5000pt jackpot)
        ent_coef=0.05,         # High curiosity to break local minima
        n_steps=2048,          # More experience per update
        batch_size=64,
        verbose=1, 
        tensorboard_log="./logs/"
    )

    print("--- DEPLOYING HRIS PRECISION MAXIMUS ---")
    try:
        model.learn(total_timesteps=10000000, callback=MaximusCallback())
        model.save("maximus_precision_final")
        task.upload_artifact("model", artifact_object="maximus_precision_final.zip")
    except KeyboardInterrupt:
        model.save("maximus_precision_interrupted")

if __name__ == "__main__":
    main()