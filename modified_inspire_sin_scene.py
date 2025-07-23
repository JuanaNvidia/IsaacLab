# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""This script demonstrates sinusoidal oscillation of a finger using the interactive scene interface.

.. code-block:: bash

    # Usage
    python modified_inspire_sin_scene.py --num_envs 1

"""

"""Launch Isaac Sim Simulator first."""

import argparse
import math
import time

from isaaclab.app import AppLauncher

# add argparse arguments
parser = argparse.ArgumentParser(description="Tutorial on sinusoidal finger oscillation.")
parser.add_argument("--num_envs", type=int, default=1, help="Number of environments to spawn.")
# append AppLauncher cli args
AppLauncher.add_app_launcher_args(parser)
# parse the arguments
args_cli = parser.parse_args()

# launch omniverse app
app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

"""Rest everything follows."""

import torch
from contextlib import suppress
from dataclasses import replace

import isaaclab.sim as sim_utils
from isaaclab.assets import ArticulationCfg, AssetBaseCfg
from isaaclab.scene import InteractiveScene, InteractiveSceneCfg
from isaaclab.sim import SimulationContext
from isaaclab.utils import configclass

# Import UI components with error handling for headless mode
with suppress(ImportError):
    import omni.ui
    import isaacsim.gui.components.ui_utils

##
# Pre-defined configs
##
# CHANGE THIS: Import from local config instead of isaaclab_assets
from robots.inspire_hand import INSPIRE_HAND_CFG


@configclass
class InspireSceneCfg(InteractiveSceneCfg):
    """Configuration for the inspire hand scene."""

    # ground plane
    ground = AssetBaseCfg(prim_path="/World/defaultGroundPlane", spawn=sim_utils.GroundPlaneCfg())

    # lights
    dome_light = AssetBaseCfg(
        prim_path="/World/Light", spawn=sim_utils.DomeLightCfg(intensity=3000.0, color=(0.75, 0.75, 0.75))
    )

    # articulation  
    inspire = replace(INSPIRE_HAND_CFG, prim_path="{ENV_REGEX_NS}/Robot")


# ... rest of the code remains the same as in the original file ... 