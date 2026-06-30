from sb3_contrib import MaskablePPO
import torch
torch.cuda.set_device(0)
def create_ppo(env, config, tensorboard_log="./logs/ppo"):
    print(f"CUDA available: {torch.cuda.is_available()}")
    model = MaskablePPO(
        policy="MultiInputPolicy",
        env=env,
        learning_rate=config["actor_lr"]["base"],
        n_steps=config["rollout_length"],
        batch_size=config["batch_size"],
        n_epochs=config["rollout_epochs"],
        gae_lambda=config["lam"],
        clip_range=config["epsilon"],
        ent_coef=config["beta"],
        gamma=config["gamma"]["base"],
        vf_coef=1,
        max_grad_norm=0.5,
        tensorboard_log=tensorboard_log,
        device="cuda",
        verbose=1
    )
    return model
