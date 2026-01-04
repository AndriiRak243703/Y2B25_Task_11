import numpy as np
from clearml import Task
from stable_baselines3 import PPO
from hris_ot2_gym_wrapper import OT2Env

def main():
    task = Task.init(
        project_name='Mentor Group - Myrthe/Group 1', 
        task_name='hris_Addiction_Withdrawal_v1', 
        reuse_last_task_id=False
    )
    
    task.set_repo(repo='https://github.com/AndriiRak243703/Y2B25_Task_11.git', branch='hris/rl-training')
    task.set_base_docker('deanis/2023y2b-rl:latest')
    task.set_packages(['tensorboard', 'clearml', 'gymnasium', 'stable-baselines3', 'pybullet'])
    task.execute_remotely(queue_name='default')

    env = OT2Env(render=False)
    
    # --- STABILIZED ADDICTION HYPERPARAMETERS ---
    model = PPO(
        "MlpPolicy", 
        env, 
        learning_rate=5e-5,    # Slower learning to recover from the brain explosion
        n_steps=2048,
        batch_size=64,
        ent_coef=0.01,         # LOWERED: Focus on the dopamine source
        gamma=0.99,
        clip_range=0.2,
        verbose=1,
        tensorboard_log="./logs/"
    )

    print("--- DEPLOYING WITHDRAWAL MAXIMUS ---")
    model.learn(total_timesteps=10000000)
    model.save("withdrawal_maximus_final")

if __name__ == "__main__":
    main()