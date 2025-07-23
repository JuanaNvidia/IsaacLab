# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""This script demonstrates how to use the interactive scene interface to setup a scene with multiple prims.

.. code-block:: bash

    # Usage
    ./isaaclab.sh -p scripts/tutorials/02_scene/create_scene.py --num_envs 32

"""

"""Launch Isaac Sim Simulator first."""


import argparse

from isaaclab.app import AppLauncher

# add argparse arguments
parser = argparse.ArgumentParser(description="Tutorial on using the interactive scene interface.")
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
from isaaclab_assets.robots.inspire_hand import INSPIRE_HAND_CFG


@configclass
class InspireSceneCfg(InteractiveSceneCfg):
    """Configuration for a cart-pole scene."""

    # ground plane
    ground = AssetBaseCfg(prim_path="/World/defaultGroundPlane", spawn=sim_utils.GroundPlaneCfg())

    # lights
    dome_light = AssetBaseCfg(
        prim_path="/World/Light", spawn=sim_utils.DomeLightCfg(intensity=3000.0, color=(0.75, 0.75, 0.75))
    )

    # articulation  
    inspire = replace(INSPIRE_HAND_CFG, prim_path="{ENV_REGEX_NS}/Robot")


class JointControlUI:
    """UI class for controlling joint positions with sliders."""
    
    def __init__(self, num_joints: int = 12):
        self.num_joints = num_joints
        self.joint_values = [0.0] * num_joints  # Default values
        self.velocity_target = 0.0  # Global velocity target
        self.effort_target = 0.0   # Global effort target
        self.sliders = []
        self.ui_window = None
        
        # Define which joints have sliders and their labels
        self.controlled_joints = {
            0: "Index",
            1: "Middle", 
            2: "Pinky",
            3: "Ring",
            4: "Thumb Yaw",
            9: "Thumb Pitch"  # Joint 10 (0-indexed)
        }
        
        self._create_ui()
    
    def _create_ui(self):
        """Create the UI window with sliders for joint control."""
        try:
            # Create UI window
            self.ui_window = omni.ui.Window("Joint Position Control", width=400, height=600)
            
            with self.ui_window.frame:
                with omni.ui.VStack(spacing=5):
                    omni.ui.Label("Joint Position Control", height=30, 
                                style={"font_size": 16, "font_weight": "bold"})
                    omni.ui.Separator()
                    
                    # Create sliders only for controlled joints
                    for joint_idx, joint_label in self.controlled_joints.items():
                        with omni.ui.HStack(height=25):
                            omni.ui.Label(f"{joint_label}:", width=80)
                            
                            # Set different ranges for different joints
                            if joint_idx == 4:  # Thumb Yaw (joint 5)
                                min_val = 0.5
                                max_val = 1.4
                                default_val = 0.5
                            elif joint_idx == 9:  # Thumb Pitch(joint 10)
                                min_val = 0.0
                                max_val = .9
                                default_val = 0.0
                            else:
                                min_val = 0.0
                                max_val = 1.5
                                default_val = 0.0
                            
                            # Update default value if needed
                            if self.joint_values[joint_idx] == 0.0:
                                self.joint_values[joint_idx] = default_val
                            
                            # Create slider model
                            slider_model = omni.ui.SimpleFloatModel()
                            slider_model.set_value(self.joint_values[joint_idx])
                            
                            # Create slider with appropriate range
                            slider = omni.ui.FloatSlider(
                                model=slider_model,
                                min=min_val,
                                max=max_val,
                                step=0.05,
                                width=200
                            )
                            
                            # Add callback to update joint value
                            def update_joint_value(model, joint_idx=joint_idx):
                                self.joint_values[joint_idx] = model.as_float
                            
                            slider_model.add_value_changed_fn(update_joint_value)
                            
                            # Value display
                            value_label = omni.ui.Label(f"{self.joint_values[joint_idx]:.2f}", width=50)
                            
                            # Update label when slider changes
                            def update_label(model, label=value_label):
                                label.text = f"{model.as_float:.2f}"
                            
                            slider_model.add_value_changed_fn(update_label)
                            
                            self.sliders.append(slider)
                    
                    omni.ui.Separator()
                    omni.ui.Label("Global Controls", height=25, 
                                style={"font_size": 14, "font_weight": "bold"})
                    
                    # Global Velocity Control
                    with omni.ui.HStack(height=25):
                        omni.ui.Label("Velocity Target:", width=80)
                        
                        velocity_model = omni.ui.SimpleFloatModel()
                        velocity_model.set_value(self.velocity_target)
                        
                        velocity_slider = omni.ui.FloatSlider(
                            model=velocity_model,
                            min=0,
                            max=100.0,
                            step=1,
                            width=200
                        )
                        
                        def update_velocity(model):
                            self.velocity_target = model.as_float
                        
                        velocity_model.add_value_changed_fn(update_velocity)
                        
                        velocity_label = omni.ui.Label(f"{self.velocity_target:.2f}", width=50)
                        
                        def update_velocity_label(model, label=velocity_label):
                            label.text = f"{model.as_float:.2f}"
                        
                        velocity_model.add_value_changed_fn(update_velocity_label)
                    
                    # Global Effort Control
                    with omni.ui.HStack(height=25):
                        omni.ui.Label("Effort Target:", width=80)
                        
                        effort_model = omni.ui.SimpleFloatModel()
                        effort_model.set_value(self.effort_target)
                        
                        effort_slider = omni.ui.FloatSlider(
                            model=effort_model,
                            min=0,
                            max=.02,
                            step=0.001,
                            width=200
                        )
                        
                        def update_effort(model):
                            self.effort_target = model.as_float
                        
                        effort_model.add_value_changed_fn(update_effort)
                        
                        effort_label = omni.ui.Label(f"{self.effort_target:.1f}", width=50)
                        
                        def update_effort_label(model, label=effort_label):
                            label.text = f"{model.as_float:.1f}"
                        
                        effort_model.add_value_changed_fn(update_effort_label)
                    
                    omni.ui.Separator()
                    omni.ui.Label("Position Controls:", style={"color": 0xFF888888})
                    omni.ui.Label("• Index, Middle, Ring, Pinky, Thumb Pitch: 0.0 to 1.5", style={"color": 0xFF888888})
                    omni.ui.Label("• Thumb Yaw: 0.5 to 1.4", style={"color": 0xFF888888})
                    omni.ui.Label("• Joints 6-9, 11-12: Fixed at 0.0", style={"color": 0xFF888888})
                    omni.ui.Label("Global Controls:", style={"color": 0xFF888888})
                    omni.ui.Label("• Velocity Target: -2.0 to 2.0 (all joints)", style={"color": 0xFF888888})
                    omni.ui.Label("• Effort Target: -10.0 to 10.0 (all joints)", style={"color": 0xFF888888})
                    
        except NameError:
            # UI components not available in headless mode
            print("[INFO]: UI components not available in headless mode. Using default values.")
    
    def get_joint_positions(self):
        """Get current joint position values as a tensor."""
        # Ensure non-controlled joints are always 0
        for i in range(self.num_joints):
            if i not in self.controlled_joints:
                self.joint_values[i] = 0.0
        return torch.tensor(self.joint_values, dtype=torch.float32)
    
    def get_joint_velocities(self):
        """Get velocity target for all joints as a tensor."""
        return torch.tensor([self.velocity_target] * self.num_joints, dtype=torch.float32)
    
    def get_joint_efforts(self):
        """Get effort target for all joints as a tensor."""
        return torch.tensor([self.effort_target] * self.num_joints, dtype=torch.float32)
    
    def cleanup(self):
        """Clean up UI resources."""
        if self.ui_window:
            self.ui_window.visible = False


def run_simulator(sim: sim_utils.SimulationContext, scene: InteractiveScene):
    """Runs the simulation loop."""
    # Extract scene entities
    robot = scene["inspire"]
    
    # Create joint control UI
    joint_ui = JointControlUI(num_joints=12)
    
    # Define simulation stepping
    sim_dt = sim.get_physics_dt()
    count = 0
    
    # Simulation loop
    while simulation_app.is_running():
        # Get current targets from UI sliders
        pos_tensor = joint_ui.get_joint_positions()
        vel_tensor = joint_ui.get_joint_velocities()
        effort_tensor = joint_ui.get_joint_efforts()
        
        # Apply targets to robot
        robot.set_joint_position_target(pos_tensor)
        robot.set_joint_velocity_target(vel_tensor)
        robot.set_joint_effort_target(effort_tensor)
        scene.write_data_to_sim()
        
        # Perform step
        sim.step()
        # Increment counter
        count += 1
        # Update buffers
        scene.update(sim_dt)
    
    # Cleanup UI
    joint_ui.cleanup()


def main():
    """Main function."""
    # Load kit helper
    sim_cfg = sim_utils.SimulationCfg(device=args_cli.device)
    sim = SimulationContext(sim_cfg)
    # Set main camera
    sim.set_camera_view((2.5, 2.5, 4.0), (0.0, 0.0, 2.0))
    # Design scene
    scene_cfg = InspireSceneCfg(num_envs=args_cli.num_envs, env_spacing=2.0)
    scene = InteractiveScene(scene_cfg)
    # Play the simulator
    sim.reset()
    # Now we are ready!
    print("[INFO]: Setup complete...")
    # Run the simulator
    run_simulator(sim, scene)


if __name__ == "__main__":
    # run the main function
    main()
    # close sim app
    simulation_app.close()
