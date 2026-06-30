import argparse
from dataclasses import dataclass

from src.gym import PathPlanningGymFactory
from src.base.evaluator import Evaluator
from src.trainer.ppo.sb3_ppo import create_ppo
from src.trainer.callbacks import GammaSchedule
from stable_baselines3.common.callbacks import CheckpointCallback
from utils import AbstractParams
import os
from sb3_contrib import MaskablePPO


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
    parser.add_argument("--checkpoint", default=None, help="Path to existing .zip model to resume from")
    args = parser.parse_args()


    params, args = PathPlanningParams.from_parsed_args(args)


    log_dir = params.create_folders(args)


    gym = PathPlanningGymFactory.create(
        params.gym
    )


    # get PPO config from json
    ppo_config = params.trainer["params"]


    # gamma scheduler callback
    gamma_callback = GammaSchedule(
       base=ppo_config["gamma"]["base"],
       target=0.99,
       decay_steps=ppo_config["gamma"]["decay_steps"],
       verbose=0
    )

    ## load model form checkpoint if specified
    checkpoint = args.checkpoint
    if checkpoint and os.path.exists(checkpoint):
        print(f"Loading checkpoint: {checkpoint}")
        model = MaskablePPO.load(checkpoint, env=gym, tensorboard_log=log_dir)
    else:
        print("Creating new PPO model...")
        model = create_ppo(gym, ppo_config, tensorboard_log=log_dir)



    
    checkpoint_callback = CheckpointCallback(
        save_freq=200_000,
        save_path=log_dir,
        name_prefix="ppo_model"
    )

    model.learn(
        total_timesteps=ppo_config["training_steps"],
        callback=[gamma_callback, checkpoint_callback]
    )


    model.save(
        log_dir + "/ppo_model"
    )


    gym.close()