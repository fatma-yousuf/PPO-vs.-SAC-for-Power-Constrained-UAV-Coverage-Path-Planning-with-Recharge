from sb3_contrib import MaskablePPO
from stable_baselines3.common.vec_env import SubprocVecEnv, DummyVecEnv
import torch

def make_env_fn(env_factory, rank):

    def _init():
        return env_factory(rank)
    return _init


if torch.cuda.is_available():
    torch.cuda.set_device(0)
    device = "cuda"
else:
    device = "cpu"
def create_ppo(env, config,env_factory=None):

    n_envs = config["rollout_gyms"]

    if env_factory is not None and n_envs > 1:
        fns = [make_env_fn(env_factory, i) for i in range(n_envs)]
        try:
            train_env = SubprocVecEnv(fns)
        except Exception:
            train_env = DummyVecEnv(fns)
    else:
        train_env = env

    

    model = MaskablePPO(
        policy="MultiInputPolicy",
        env=train_env,

        learning_rate=config["actor_lr"]["base"],

        n_steps=config["rollout_length"]//n_envs,

        batch_size=config["batch_size"],

        n_epochs=config["rollout_epochs"],

        gae_lambda=config["lam"],

        clip_range=config["epsilon"],

        ent_coef=config["beta"],

        gamma=config["gamma"]["base"],

        vf_coef=0.5,

        max_grad_norm=0.5,
        normalize_advantage=config.get("normalize_advantage", True),

        tensorboard_log="./logs/ppo",
        device="cuda",

        verbose=1
    )
    return model