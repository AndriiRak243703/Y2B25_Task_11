import numpy as np
import plotly.graph_objects as go
from clearml import Task
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import BaseCallback
from hris_ot2_gym_wrapper import OT2Env

class GachaCallback(BaseCallback):
    def __init__(self, log_freq=1024):
        super().__init__()
        self.log_freq = log_freq
        self.trajectory = []
        print(f"\n{'Step':<10} | {'Error (mm)':<12} | {'Dopamine Status'}")
        print("-" * 50)

    def _on_step(self) -> bool:
        obs = self.locals['new_obs'][0]
        self.trajectory.append(obs[:3])
        if self.n_calls % self.log_freq == 0:
            dist_mm = np.linalg.norm(obs[:3] - obs[3:]) * 150
            if dist_mm < 1.5: status = "💎 JACKPOT!!" 
            elif dist_mm < 15: status = "💰 BIG WINNER"
            else: status = "📉 BROKE"
            print(f"{self.num_timesteps:<10} | {dist_mm:>10.2f} mm | {status}")
            self._report_plot()
        return True

    def _report_plot(self):
        path = np.array(self.trajectory)
        fig = go.Figure(data=[go.Scatter3d(x=path[:,0], y=path[:,1], z=path[:,2], mode='lines', line=dict(color='gold', width=5))])
        Task.current_task().get_logger().report_plotly(title="3D Path", series="Trajectory", figure=fig)
        self.trajectory = []

def main():
    task = Task.init(project_name='Mentor Group - Myrthe/Group 1', task_name='hris_Addiction_Maximus_GACHA_V1', reuse_last_task_id=False)
    task.set_repo(repo='https://github.com/AndriiRak243703/Y2B25_Task_11.git', branch='hris/rl-training')
    task.set_base_docker('deanis/2023y2b-rl:latest')
    task.set_packages(['clearml', 'gymnasium', 'stable-baselines3', 'pybullet', 'plotly'])
    task.execute_remotely(queue_name='default')

    env = OT2Env(render=False)
    
    # --- HIGH GAMBLE HYPERPARAMETERS ---
    model = PPO(
        "MlpPolicy", 
        env, 
        learning_rate=2e-4,    # Faster learning for fast feedback
        n_steps=2048,
        batch_size=64,
        ent_coef=0.1,          # MAX CURIOSITY: Agent will try everything
        gamma=0.995,           # Vision: Thinks about the 10,000pt jackpot far in the future
        verbose=1,
        tensorboard_log="./logs/"
    )

    print("--- DEPLOYING GACHA MAXIMUS ---")
    model.learn(total_timesteps=4000000, callback=GachaCallback())
    model.save("gacha_maximus_final")

if __name__ == "__main__":
    main()