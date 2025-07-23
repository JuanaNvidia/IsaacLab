# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Demo script for the Inspire Hand manager-based RL environment."""

import argparse
import torch

import isaaclab.sim as sim_utils
from isaaclab.envs import ManagerBasedRLEnv
from isaaclab.utils.argparse import add_default_arguments

from .inspire_hand_env_cfg import InspireHandEnvCfg


def main():
    """Main function."""
    # parse arguments
    parser = argparse.ArgumentParser(description="Inspire Hand Environment Demo")
    add_default_arguments(parser)
    args_cli = parser.parse_args()

    # create environment configuration
    env_cfg = InspireHandEnvCfg()
    env_cfg.scene.num_envs = args_cli.num_envs
    env_cfg.sim.device = args_cli.device

    # setup RL environment
    env = ManagerBasedRLEnv(cfg=env_cfg)

    # simulate physics
    count = 0
    while sim_utils.SimulationApp.is_running():
        with torch.inference_mode():
            # reset
            if count % 300 == 0:
                count = 0
                env.reset()
                print("-" * 80)
                print("[INFO]: Resetting environment...")
            
            # sample random actions
            actions = torch.randn_like(env.action_manager.action)
            
            # step the environment
            obs, rew, terminated, truncated, info = env.step(actions)
            
            # print current object position
            if "object_pos" in obs["policy"]:
                object_pos = obs["policy"]["object_pos"][0]
                print(f"[Env 0]: Object position: {object_pos}")
            
            # print reward
            if "object_goal_dist" in info["log"]:
                reward = info["log"]["object_goal_dist"]
                print(f"[Env 0]: Distance reward: {reward:.4f}")
            
            # update counter
            count += 1

    # close the environment
    env.close()


if __name__ == "__main__":
    main() 