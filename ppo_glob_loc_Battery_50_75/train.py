import argparse
import copy
from dataclasses import dataclass
import dataclasses
import os
from datetime import datetime

from src.gym import PathPlanningGymFactory
from src.base.evaluator import Evaluator
from src.trainer.ppo.sb3_ppo import create_ppo
from src.trainer.callbacks import GammaSchedule, DualLRCallback
from src.gym_wrapper.gym_wrapper import MaskableGymWrapper

from utils import AbstractParams
from sb3_contrib import MaskablePPO
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.callbacks import CallbackList, CheckpointCallback


@dataclass
class PathPlanningParams(AbstractParams):
    trainer: dict
    gym: PathPlanningGymFactory.default_param_type() = (
        PathPlanningGymFactory.default_params()
    )
    evaluator: Evaluator.Params = Evaluator.Params()


if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser = PathPlanningParams.add_args_to_parser(parser)
    args = parser.parse_args()

    params, args = PathPlanningParams.from_parsed_args(args)

    # ========================
    # ALWAYS CREATE NEW RUN DIR
    # ========================
    base_log_dir = params.create_folders(args)

    run_id = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_dir = os.path.join(base_log_dir, f"run_{run_id}")
    os.makedirs(run_dir, exist_ok=True)

    print("Run directory:", run_dir)

    ppo_config = params.trainer["params"]
    gym_params = params.gym

    # ========================
    # ENV FACTORY
    # ========================
    def env_factory(rank=0):
        p = copy.deepcopy(gym_params)
        if dataclasses.is_dataclass(p):
            p = dataclasses.asdict(p)

        env = PathPlanningGymFactory.create(p)
        env = Monitor(env)
        return MaskableGymWrapper(env)

    env = env_factory()

    # ========================
    # CALLBACKS
    # ========================
    gamma_callback = GammaSchedule(
        base=ppo_config["gamma"]["base"],
        target=0.999,
        decay_steps=ppo_config["gamma"]["decay_steps"],
        decay_rate=ppo_config["gamma"]["decay_rate"],
        verbose=1,
    )

    dual_lr_callback = DualLRCallback(
        actor_lr_config=ppo_config["actor_lr"],
        critic_lr_config=ppo_config["critic_lr"],
        verbose=1,
    )

    checkpoint_callback = CheckpointCallback(
        save_freq=20000,
        save_path=run_dir,
        name_prefix="ppo_checkpoint"
    )

    callback_list = CallbackList([
        gamma_callback,
        dual_lr_callback,
        checkpoint_callback
    ])

    # ========================
    # RESUME LOGIC (SAFE)
    # ========================
    checkpoint_path = os.path.join(run_dir, "ppo_checkpoint.zip")

    if os.path.exists(checkpoint_path):
        print("Resuming from checkpoint...")
        model = MaskablePPO.load(checkpoint_path, env=env)
        is_resuming = True
    else:
        print("Creating new PPO model...")
        model = create_ppo(
            env,
            ppo_config,
            env_factory=env_factory
        )
        is_resuming = False

    # ========================
    # TRAIN
    # ========================
    model.learn(
        total_timesteps=ppo_config["training_steps"],
        callback=callback_list,
        reset_num_timesteps=not is_resuming,
        tb_log_name=run_id
    )

    # ========================
    # FINAL SAVE
    # ========================
    model.save(os.path.join(run_dir, "ppo_final"))

    env.close()