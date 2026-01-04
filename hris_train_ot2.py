import numpy as np
from clearml import Task
from stable_baselines3 import PPO
from hris_ot2_gym_wrapper import OT2Env

def main():
    # 1. INITIALIZE TASK (USING REUSE=FALSE TO CLEAR DOCKER ERRORS)
    task = Task.init(
        project_name='Mentor Group - Jason/Group 1', 
        task_name='hris_Addiction_Maximus_FINAL_SYNC',
        reuse_last_task_id=False
    )

    # 2. REMOTE CONNECTION (YOUR FRIEND'S LOGIC)
    task.set_repo(
        repo='https://github.com/AndriiRak243703/Y2B25_Task_11.git',
        branch='hris/rl-training'
    )
    task.set_base_docker('deanis/2023y2b-rl:latest')
    task.set_packages(['tensorboard', 'clearml', 'gymnasium', 'stable-baselines3', 'pybullet'])

    # Enqueue for remote GPU execution
    task.execute_remotely(queue_name='default')

    # 3. TRAINING DEPLOYMENT
    env = OT2Env(render=False)
    
    model = PPO(
        "MlpPolicy", 
        env, 
        learning_rate=5e-5,
        gamma=0.98,
        verbose=1,
        tensorboard_log="./logs/"
    )

    print("--- DEPLOYING HRIS ADDICTION MAXIMUS ---")
    model.learn(total_timesteps=2000000)
    
    model.save("hris_maximus_final_model")
    task.upload_artifact("model", artifact_object="hris_maximus_final_model.zip")

if __name__ == "__main__":
    main()