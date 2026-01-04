from clearml import Task
from stable_baselines3 import PPO
from hris_ot2_gym_wrapper import OT2Env

def main():
    # ClearML Integration
    task = Task.init(
        project_name='Mentor Group - Myrthe/Group 1', 
        task_name='hris_PPO_Precision_Final',
        reuse_last_task_id=False
    )
    
    # Remote execution settings
    task.set_base_docker('deanis/2023y2b-rl:latest')
    task.execute_remotely(queue_name='default')

    env = OT2Env(render=False)

    # Requested PPO Configuration
    model = PPO(
        "MlpPolicy",
        env,
        learning_rate=3e-4,
        n_steps=2048,
        batch_size=64,
        n_epochs=10,
        gamma=0.995,        # High Gamma for long-horizon planning
        gae_lambda=0.95,
        clip_range=0.2,
        ent_coef=0.001,     # Low entropy forces precision near the goal
        vf_coef=0.5,
        max_grad_norm=0.5,
        verbose=1,
        tensorboard_log="./ppo_tensorboard/"
    )

    print("--- DEPLOYING PRECISION MAXIMUS ---")
    model.learn(total_timesteps=4000000)
    
    # Save the model
    model.save("ppo_ot2_precision_model")
    task.upload_artifact("model", "ppo_ot2_precision_model.zip")

if __name__ == "__main__":
    main()