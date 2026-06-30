import argparse
import json
import numpy as np
from stable_baselines3.common import base_class          

from src.gym import PathPlanningGymFactory
from src.gym_wrapper.gym_wrapper import MaskableGymWrapper
from stable_baselines3.common.monitor import Monitor
from sb3_contrib import MaskablePPO


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('config', help='Path to config file')
    parser.add_argument('--model', required=True, help='Path to model zip file')
    parser.add_argument('--episodes', default=10, type=int, help='Number of episodes to evaluate')
    parser.add_argument('--render', action='store_true', help='Render environment')
    args = parser.parse_args()

    with open(args.config) as f:
        params = json.load(f)

    gym_params = params["gym"]

    if args.render:
        gym_params["params"]["rendering"]["render"] = True

    env = PathPlanningGymFactory.create(gym_params)
    env = Monitor(env)
    env = MaskableGymWrapper(env)
    
    _orig = base_class.BaseAlgorithm.set_parameters
    def _patched(self, load_path_or_dict, exact_match=True, device="auto"):
        if isinstance(load_path_or_dict, dict):
            load_path_or_dict = {k: v for k, v in load_path_or_dict.items()
                                 if "optimizer" not in k}
        return _orig(self, load_path_or_dict, exact_match=False, device=device)
    base_class.BaseAlgorithm.set_parameters = _patched

    model = MaskablePPO.load(
        args.model,
        env=env,
        device="cpu",                                    
        custom_objects={
            "observation_space": env.observation_space,
            "action_space": env.action_space,
            "lr_schedule": lambda _: 3e-5,              
        }
    )

    ep_rewards = []
    ep_lengths = []
    ep_collection_ratios = []

    for ep in range(args.episodes):
        obs, _ = env.reset()
        done = False
        truncated = False
        total_reward = 0
        steps = 0
        info = {}

        while not done and not truncated:
            action, _ = model.predict(
                obs,
                action_masks=env.action_masks(),
                deterministic=True
            )
            action = np.array([int(action)])   # fix 0-dim scalar
            obs, reward, done, truncated, info = env.step(action)
            total_reward += reward
            steps += 1

        ep_rewards.append(total_reward)
        ep_lengths.append(steps)
        ep_collection_ratios.append(info.get("collection_ratio", 0.0))
        print(f"Episode {ep+1}: reward={total_reward:.2f}, steps={steps}, "
              f"coverage={info.get('collection_ratio', 0.0):.2%}, "
              f"solved={info.get('task_solved', False)}")

    print(f"\n--- Results over {args.episodes} episodes ---")
    print(f"Mean reward:           {sum(ep_rewards)/len(ep_rewards):.2f}")
    print(f"Mean length:           {sum(ep_lengths)/len(ep_lengths):.1f}")
    print(f"Mean coverage ratio:   {sum(ep_collection_ratios)/len(ep_collection_ratios):.2%}")

    env.close()


if __name__ == "__main__":
    main()