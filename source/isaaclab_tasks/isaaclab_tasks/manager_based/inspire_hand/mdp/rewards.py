# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from __future__ import annotations

import torch
from typing import TYPE_CHECKING

from isaaclab.assets import RigidObject
from isaaclab.managers import SceneEntityCfg
from isaaclab.utils.math import quat_conjugate, quat_mul

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv


def object_goal_distance(
    env: ManagerBasedRLEnv,
    std: float,
    object_cfg: SceneEntityCfg = SceneEntityCfg("object"),
    goal_cfg: SceneEntityCfg = SceneEntityCfg("goal_marker"),
) -> torch.Tensor:
    """Reward the agent for moving the object close to the goal using tanh-kernel."""
    # extract the used quantities (to enable type-hinting)
    object: RigidObject = env.scene[object_cfg.name]
    goal: RigidObject = env.scene[goal_cfg.name]
    
    # distance of the object to the goal: (num_envs,)
    distance = torch.norm(object.data.root_pos_w - goal.data.root_pos_w, dim=1)
    
    return 1 - torch.tanh(distance / std)


def object_goal_orientation(
    env: ManagerBasedRLEnv,
    std: float,
    object_cfg: SceneEntityCfg = SceneEntityCfg("object"),
    goal_cfg: SceneEntityCfg = SceneEntityCfg("goal_marker"),
) -> torch.Tensor:
    """Reward the agent for aligning the object orientation with the goal using tanh-kernel."""
    # extract the used quantities (to enable type-hinting)
    object: RigidObject = env.scene[object_cfg.name]
    goal: RigidObject = env.scene[goal_cfg.name]
    
    # orientation difference
    quat_diff = quat_mul(object.data.root_quat_w, quat_conjugate(goal.data.root_quat_w))
    # compute rotation distance (angle in radians)
    rot_dist = 2.0 * torch.asin(torch.clamp(torch.norm(quat_diff[:, 1:4], p=2, dim=-1), max=1.0))
    
    return 1 - torch.tanh(rot_dist / std)


def is_success(
    env: ManagerBasedRLEnv,
    tolerance: float,
    object_cfg: SceneEntityCfg = SceneEntityCfg("object"),
    goal_cfg: SceneEntityCfg = SceneEntityCfg("goal_marker"),
) -> torch.Tensor:
    """Reward for successful completion of the task."""
    # extract the used quantities (to enable type-hinting)
    object: RigidObject = env.scene[object_cfg.name]
    goal: RigidObject = env.scene[goal_cfg.name]
    
    # position distance
    pos_dist = torch.norm(object.data.root_pos_w - goal.data.root_pos_w, dim=1)
    
    # orientation difference
    quat_diff = quat_mul(object.data.root_quat_w, quat_conjugate(goal.data.root_quat_w))
    rot_dist = 2.0 * torch.asin(torch.clamp(torch.norm(quat_diff[:, 1:4], p=2, dim=-1), max=1.0))
    
    # success if both position and orientation are within tolerance
    success = (pos_dist <= tolerance) & (rot_dist <= tolerance)
    
    return success.float()


def object_height_below_threshold(
    env: ManagerBasedRLEnv,
    height_threshold: float,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("object"),
) -> torch.Tensor:
    """Penalize when the object falls below a certain height."""
    # extract the used quantities (to enable type-hinting)
    asset: RigidObject = env.scene[asset_cfg.name]
    return (asset.data.root_pos_w[:, 2] < height_threshold).float() 