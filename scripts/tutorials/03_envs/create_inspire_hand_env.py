# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""
This script demonstrates how to create a manager-based environment for the Inspire Hand.
It shows the basic setup and usage of the environment.

.. code-block:: bash

    ./isaaclab.sh -p scripts/tutorials/03_envs/create_inspire_hand_env.py --num_envs 32

"""

"""Launch Isaac Sim Simulator first."""

import argparse

from isaaclab.app import AppLauncher

# add argparse arguments
parser = argparse.ArgumentParser(description="Tutorial on creating an Inspire Hand environment.")
parser.add_argument("--num_envs", type=int, default=16, help="Number of environments to spawn.")

# append AppLauncher cli args
AppLauncher.add_app_launcher_args(parser)
# parse the arguments
args_cli = parser.parse_args()

# launch omniverse app
app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

"""Rest everything follows."""

import torch

from isaaclab.envs import ManagerBasedRLEnv
from isaaclab_tasks.manager_based.inspire_hand import InspireHandEnvCfg


def main():
    """Main function."""
    # parse the arguments
    env_cfg = InspireHandEnvCfg()
    env_cfg.scene.num_envs = args_cli.num_envs
    env_cfg.sim.device = args_cli.device
    
    # setup environment
    env = ManagerBasedRLEnv(cfg=env_cfg)

    # simulate physics
    count = 0
    while simulation_app.is_running():
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
            obs, rewards, terminated, truncated, info = env.step(actions)
            
            # print current object position and reward
            print(f"[Env 0]: Object pos: {obs['policy'][0][:3].cpu().numpy()}")
            # Handle different reward structures
            try:
                if isinstance(rewards, dict):
                    if 'policy' in rewards:
                        reward_value = rewards['policy'][0].item()
                    else:
                        reward_value = list(rewards.values())[0][0].item()
                else:
                    reward_value = rewards[0].item()
                print(f"[Env 0]: Reward: {reward_value:.3f}")
            except:
                print(f"[Env 0]: Reward: {rewards}")
            
            # update counter
            count += 1

    # close the environment
    env.close()


if __name__ == "__main__":
    # run the main function
    main()
    # close sim app
    simulation_app.close() 