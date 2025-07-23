# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from __future__ import annotations

from isaaclab.envs import ManagerBasedRLEnv

from .inspire_hand_env_cfg import InspireHandEnvCfg


class InspireHandEnv(ManagerBasedRLEnv):
    """Manager-based RL environment for Inspire Hand in-hand manipulation."""

    cfg: InspireHandEnvCfg

    def __init__(self, cfg: InspireHandEnvCfg, render_mode: str | None = None, **kwargs):
        """Initialize the environment.

        Args:
            cfg: Configuration for the environment.
            render_mode: Rendering mode for the environment.
            **kwargs: Additional arguments.
        """
        super().__init__(cfg, render_mode, **kwargs) 