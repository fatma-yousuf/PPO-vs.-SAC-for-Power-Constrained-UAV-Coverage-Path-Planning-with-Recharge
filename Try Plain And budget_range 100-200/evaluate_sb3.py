import argparse
from dataclasses import dataclass

import numpy as np

from sb3_contrib import MaskablePPO

from src.gym import PathPlanningGymFactory
from src.base.evaluator import Evaluator
from utils import AbstractParams


@dataclass
class PathPlanningParams(AbstractParams):
    trainer: dict
    gym: PathPlanningGymFactory.default_param_type() = (
        PathPlanningGymFactory.default_params()
    )
    evaluator: Evaluator.Params = Evaluator.Params()


def main():
    parser = argparse.ArgumentParser(description="Evaluate a trained SB3 MaskablePPO model")
    parser = PathPlanningParams.add_args_to_parser(parser)
    parser.add_argument("--model", required=True, help="Path to saved model .zip (e.g. logs/.../ppo_model.zip)")
    parser.add_argument("--episodes", type=int, default=10, help="Number of episodes to run (default: 10)")
    parser.add_argument("--render", action="store_true", help="Enable visual rendering")
    args = parser.parse_args()

    # Load params from config
    params, args = PathPlanningParams.from_parsed_args(args)

    # Enable rendering if requested
    if args.render:
        params.gym["params"]["rendering"]["render"] = True

    # Create environment
    print("\nLoading environment...")
    gym = PathPlanningGymFactory.create(params.gym)

    # Load SB3 model
    print(f"Loading model: {args.model}")
    model = MaskablePPO.load(args.model, env=gym)
    print("Model loaded.\n")

    # Evaluation loop
    episode_rewards   = []
    episode_lengths   = []
    tasks_solved      = 0
    collection_ratios = []
    crashed_count     = 0
    boundary_counts   = []

    print(f"Running {args.episodes} episodes...")
    print("-" * 70)

    for ep in range(args.episodes):
        obs, info = gym.reset()
        done = False
        total_reward = 0.0

        while not done:
            action_masks = gym.action_masks()
            action, _ = model.predict(obs, action_masks=action_masks, deterministic=True)
            action = int(np.asarray(action).flat[0])  # unwrap to plain int
            obs, reward, terminated, truncated, info = gym.step(action)
            total_reward += float(reward)
            done = terminated or truncated

        # info keys from GridGym.get_info + CPPGym.get_info:
        # landed, crashed, terminal, boundary_counter, episodic_reward,
        # task_solved, total_steps, timeout, charging_steps,
        # map_index, map_name, collection_ratio, completion_steps (only if solved)
        solved    = info.get("task_solved", False)
        col_ratio = info.get("collection_ratio", 0.0)
        steps     = info.get("total_steps", 0)
        crashed   = info.get("crashed", False)
        boundary  = info.get("boundary_counter", 0)
        charging  = info.get("charging_steps", 0)
        timeout   = info.get("timeout", False)
        map_name  = info.get("map_name", "?")

        episode_rewards.append(total_reward)
        episode_lengths.append(steps)
        collection_ratios.append(col_ratio)
        boundary_counts.append(boundary)
        if solved:
            tasks_solved += 1
        if crashed:
            crashed_count += 1

        status = "SOLVED " if solved else ("CRASHED" if crashed else ("TIMEOUT" if timeout else "       "))
        print(
            f"Ep {ep+1:2d} | {status} | "
            f"Reward: {total_reward:8.3f} | "
            f"Steps: {steps:5d} | "
            f"Coverage: {col_ratio:5.1%} | "
            f"Charges: {charging:3d} | "
            f"Map: {map_name}"
        )

    # Summary
    n = args.episodes
    print("-" * 70)
    print(f"\nSummary over {n} episodes:")
    print(f"  Mean Reward   : {np.mean(episode_rewards):8.3f}  +/-  {np.std(episode_rewards):.3f}")
    print(f"  Mean Steps    : {np.mean(episode_lengths):8.1f}")
    print(f"  Mean Coverage : {np.mean(collection_ratios):8.1%}")
    print(f"  Tasks Solved  : {tasks_solved}/{n}  ({tasks_solved/n:.0%})")
    print(f"  Crashes       : {crashed_count}/{n}")
    print(f"  Mean Boundary : {np.mean(boundary_counts):8.1f}  (infeasible action attempts)")
    print(f"  Best Reward   : {np.max(episode_rewards):8.3f}")
    print(f"  Worst Reward  : {np.min(episode_rewards):8.3f}")

    gym.close()


if __name__ == "__main__":
    main()
