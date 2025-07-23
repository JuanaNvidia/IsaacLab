# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Configuration for the Inspire Hand robot."""

import isaaclab.sim as sim_utils
from isaaclab.actuators.actuator_cfg import ImplicitActuatorCfg
from isaaclab.assets.articulation import ArticulationCfg

##
# Configuration
##

INSPIRE_HAND_CFG = ArticulationCfg(
    spawn=sim_utils.UsdFileCfg(
        # Update this path to your local USD file
        usd_path="./assets/inspire_hand_left.usd",  # Relative path in your repo
        activate_contact_sensors=False,
        rigid_props=sim_utils.RigidBodyPropertiesCfg(
            disable_gravity=True,
            retain_accelerations=True,
            max_depenetration_velocity=1000.0,
        ),
        articulation_props=sim_utils.ArticulationRootPropertiesCfg(
            enabled_self_collisions=True,
            solver_position_iteration_count=8,
            solver_velocity_iteration_count=0,
            sleep_threshold=0.005,
            stabilization_threshold=0.0005,
        ),
        joint_drive_props=sim_utils.JointDrivePropertiesCfg(drive_type="force"),
        fixed_tendons_props=sim_utils.FixedTendonPropertiesCfg(limit_stiffness=30.0, damping=0.1),
    ),
    init_state=ArticulationCfg.InitialStateCfg(
        pos=(0.0, 0.0, 0.5),
        rot=(0.0, 0.7071, 0.0, 0.7071),
        joint_pos={".*": 0.0},
    ),
    actuators={
        "fingers": ImplicitActuatorCfg(
            joint_names_expr=[
                "index_proximal_joint", "index_intermediate_joint", 
                "middle_proximal_joint", "middle_intermediate_joint",
                "pinky_proximal_joint", "pinky_intermediate_joint",
                "ring_proximal_joint", "ring_intermediate_joint",
                "thumb_proximal_yaw_joint", "thumb_proximal_pitch_joint", 
                "thumb_intermediate_joint", "thumb_distal_joint",
            ],
            effort_limit=None,
            stiffness=None,
            damping=None,
        ),
    },
    soft_joint_pos_limit_factor=1.0,
) 