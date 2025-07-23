# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""This script demonstrates sinusoidal oscillation of a finger using the interactive scene interface.

.. code-block:: bash

    # Usage
    ./isaaclab.sh -p scripts/tutorials/02_scene/inspire_sin_scene.py --num_envs 1

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
# args_cli.experience = "isaacsim.exp.full.kit"

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
    """Configuration for the inspire hand scene."""

    # ground plane
    ground = AssetBaseCfg(prim_path="/World/defaultGroundPlane", spawn=sim_utils.GroundPlaneCfg())

    # lights
    dome_light = AssetBaseCfg(
        prim_path="/World/Light", spawn=sim_utils.DomeLightCfg(intensity=3000.0, color=(0.75, 0.75, 0.75))
    )

    # articulation  
    inspire = replace(INSPIRE_HAND_CFG, prim_path="{ENV_REGEX_NS}/Robot")


class SinusoidalFingerUI:
    """UI class for controlling sinusoidal finger oscillation."""
    
    def __init__(self, num_joints: int = 12):
        self.num_joints = num_joints
        self.joint_values = [0.75] * (num_joints - 2) + [0.0] * 2
        self.oscillation_speed = 1.0  # Hz
        self.oscillation_amplitude = 0.5  # Range of motion
        self.oscillation_offset = 0.75  # Center position
        self.effort_target = 0.0  # Global effort target
        self.ui_window = None
        self.start_time = time.time()
        
        # Define finger options
        self.finger_options = {
            "Index": 0,
            "Middle": 1,
            "Ring": 2,
            "Pinky": 3,
            "Thumb Yaw": 4,
            "Thumb Pitch": 9
        }
        self.selected_finger = 0  # Default to Index finger
        
        self._create_ui()
    
    def _create_ui(self):
        """Create the UI window with controls for sinusoidal oscillation."""
        try:
            # Create UI window
            self.ui_window = omni.ui.Window("Sinusoidal Finger Control", width=450, height=400)
            
            with self.ui_window.frame:
                with omni.ui.VStack(spacing=5):
                    omni.ui.Label("Sinusoidal Finger Oscillation", height=30, 
                                style={"font_size": 16, "font_weight": "bold"})
                    omni.ui.Separator()
                    
                    # Finger Selection
                    with omni.ui.HStack(height=25):
                        omni.ui.Label("Finger:", width=100)
                        
                        finger_combo = omni.ui.ComboBox(
                            self.selected_finger,
                            *list(self.finger_options.keys()),
                            width=150
                        )
                        
                        def update_finger_selection(model, item):
                            finger_name = list(self.finger_options.keys())[model.get_item_value_model().as_int]
                            self.selected_finger = self.finger_options[finger_name]
                        
                        finger_combo.model.add_item_changed_fn(update_finger_selection)
                    
                    omni.ui.Spacer(height=10)
                    
                    # Oscillation Speed Control
                    with omni.ui.HStack(height=25):
                        omni.ui.Label("Speed (Hz):", width=100)
                        
                        speed_model = omni.ui.SimpleFloatModel()
                        speed_model.set_value(self.oscillation_speed)
                        
                        speed_slider = omni.ui.FloatSlider(
                            model=speed_model,
                            min=0.1,
                            max=5.0,
                            step=0.1,
                            width=200
                        )
                        
                        def update_speed(model):
                            self.oscillation_speed = model.as_float
                        
                        speed_model.add_value_changed_fn(update_speed)
                        
                        speed_label = omni.ui.Label(f"{self.oscillation_speed:.1f}", width=50)
                        
                        def update_speed_label(model, label=speed_label):
                            label.text = f"{model.as_float:.1f}"
                        
                        speed_model.add_value_changed_fn(update_speed_label)
                    
                    # Oscillation Amplitude Control
                    with omni.ui.HStack(height=25):
                        omni.ui.Label("Amplitude:", width=100)
                        
                        amplitude_model = omni.ui.SimpleFloatModel()
                        amplitude_model.set_value(self.oscillation_amplitude)
                        
                        amplitude_slider = omni.ui.FloatSlider(
                            model=amplitude_model,
                            min=0.1,
                            max=1.0,
                            step=0.05,
                            width=200
                        )
                        
                        def update_amplitude(model):
                            self.oscillation_amplitude = model.as_float
                        
                        amplitude_model.add_value_changed_fn(update_amplitude)
                        
                        amplitude_label = omni.ui.Label(f"{self.oscillation_amplitude:.2f}", width=50)
                        
                        def update_amplitude_label(model, label=amplitude_label):
                            label.text = f"{model.as_float:.2f}"
                        
                        amplitude_model.add_value_changed_fn(update_amplitude_label)
                    
                    # Center Position Control
                    with omni.ui.HStack(height=25):
                        omni.ui.Label("Center Pos:", width=100)
                        
                        offset_model = omni.ui.SimpleFloatModel()
                        offset_model.set_value(self.oscillation_offset)
                        
                        offset_slider = omni.ui.FloatSlider(
                            model=offset_model,
                            min=0.0,
                            max=1.5,
                            step=0.05,
                            width=200
                        )
                        
                        def update_offset(model):
                            self.oscillation_offset = model.as_float
                        
                        offset_model.add_value_changed_fn(update_offset)
                        
                        offset_label = omni.ui.Label(f"{self.oscillation_offset:.2f}", width=50)
                        
                        def update_offset_label(model, label=offset_label):
                            label.text = f"{model.as_float:.2f}"
                        
                        offset_model.add_value_changed_fn(update_offset_label)
                    
                    # Effort Control
                    with omni.ui.HStack(height=25):
                        omni.ui.Label("Effort:", width=100)
                        
                        effort_model = omni.ui.SimpleFloatModel()
                        effort_model.set_value(self.effort_target)
                        
                        effort_slider = omni.ui.FloatSlider(
                            model=effort_model,
                            min=0,
                            max=0.02,
                            step=0.001,
                            width=200
                        )
                        
                        def update_effort(model):
                            self.effort_target = model.as_float
                        
                        effort_model.add_value_changed_fn(update_effort)
                        
                        effort_label = omni.ui.Label(f"{self.effort_target:.3f}", width=50)
                        
                        def update_effort_label(model, label=effort_label):
                            label.text = f"{model.as_float:.3f}"
                        
                        effort_model.add_value_changed_fn(update_effort_label)
                    
                    omni.ui.Separator()
                    
                    # Reset button
                    with omni.ui.HStack(height=30):
                        omni.ui.Spacer()
                        reset_button = omni.ui.Button("Reset Timer", width=100)
                        
                        def reset_timer():
                            self.start_time = time.time()
                        
                        reset_button.set_clicked_fn(reset_timer)
                        omni.ui.Spacer()
                    
                    omni.ui.Separator()
                    omni.ui.Label("Controls:", style={"color": 0xFF888888})
                    omni.ui.Label("• Speed: Oscillation frequency in Hz", style={"color": 0xFF888888})
                    omni.ui.Label("• Amplitude: Range of motion around center", style={"color": 0xFF888888})
                    omni.ui.Label("• Center Pos: Center position for oscillation", style={"color": 0xFF888888})
                    omni.ui.Label("• Effort: Applied force/torque to all joints", style={"color": 0xFF888888})
                    
        except NameError:
            # UI components not available in headless mode
            print("[INFO]: UI components not available in headless mode. Using default values.")
    
    def get_joint_positions(self):
        """Get current joint position values with sinusoidal oscillation."""
        current_time = time.time() - self.start_time
        
        # Calculate sinusoidal position for selected finger
        sin_value = math.sin(2 * math.pi * self.oscillation_speed * current_time)
        oscillating_position = self.oscillation_offset + self.oscillation_amplitude * sin_value
        
        # Set all joints to 0 except the selected finger
        for i in range(self.num_joints):
            if i == self.selected_finger:
                self.joint_values[i] = oscillating_position
            else:
                self.joint_values[i] = 0.75
                
        return torch.tensor(self.joint_values, dtype=torch.float32)
    
    def get_joint_efforts(self):
        """Get effort target for all joints as a tensor."""
        return torch.tensor([self.effort_target] * self.num_joints, dtype=torch.float32)
    
    def cleanup(self):
        """Clean up UI resources."""
        if self.ui_window:
            self.ui_window.visible = False


def run_simulator(sim: sim_utils.SimulationContext, scene: InteractiveScene):
    """Runs the simulation loop with sinusoidal finger oscillation."""
    # Extract scene entities
    robot = scene["inspire"]
    
    # Create sinusoidal finger control UI
    finger_ui = SinusoidalFingerUI(num_joints=12)
    
    # Define simulation stepping
    sim_dt = sim.get_physics_dt()
    count = 0
    
    # Joint names for reference
    joint_names = [
        "index_proximal", "middle_proximal", "pinky_proximal", "ring_proximal",
        "thumb_yaw", "joint_5", "joint_6", "joint_7", "joint_8", "thumb_pitch",
        "joint_10", "joint_11"
    ]
    
    # Print header
    print("\n" + "="*80)
    print("JOINT POSITIONS MONITOR")
    print("="*80)
    print(f"{'Step':>5} | {'Selected':>8} | " + " | ".join([f"{name:>8}" for name in joint_names]))
    print("-" * 80)
    
    # Simulation loop
    while simulation_app.is_running():
        # Get current targets from UI
        pos_tensor = finger_ui.get_joint_positions()
        effort_tensor = finger_ui.get_joint_efforts()
        
        # Apply targets to robot
        robot.set_joint_position_target(pos_tensor)
        robot.set_joint_effort_target(effort_tensor)
        scene.write_data_to_sim()
        
        # Perform step
        sim.step()
        # Increment counter
        count += 1
        # Update buffers
        scene.update(sim_dt)
        
        # Print joint positions every 30 steps (0.5 seconds at 60Hz)
        if count % 30 == 0:
            # Get current joint positions from the robot
            current_joint_pos = robot.data.joint_pos[0]  # [0] for first environment
            
            # Get selected finger name
            finger_names = list(finger_ui.finger_options.keys())
            selected_finger_name = finger_names[list(finger_ui.finger_options.values()).index(finger_ui.selected_finger)]
            
            # Format and print the joint positions
            pos_str = " | ".join([f"{pos:.3f}" for pos in current_joint_pos])
            print(f"{count:>5} | {selected_finger_name:>8} | {pos_str}")
    
    # Cleanup UI
    finger_ui.cleanup()


def main():
    """Main function."""
    # Load kit helper
    sim_cfg = sim_utils.SimulationCfg(device=args_cli.device, dt=0.02)  # 50 Hz = 1/50 = 0.02 seconds/
    sim = SimulationContext(sim_cfg)
    # Set main camera
    sim.set_camera_view((1.5, 1.5, 4.0), (0.0, 0.0, 2.0))
    # Design scene
    scene_cfg = InspireSceneCfg(num_envs=args_cli.num_envs, env_spacing=2.0)
    scene = InteractiveScene(scene_cfg)
    # Play the simulator
    sim.reset()
    # Now we are ready!
    print("[INFO]: Setup complete...")
    print("[INFO]: Selected finger will oscillate sinusoidally")
    # Run the simulator
    run_simulator(sim, scene)


if __name__ == "__main__":
    # run the main function
    main()
    # close sim app
    simulation_app.close()
