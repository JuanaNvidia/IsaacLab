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
parser.add_argument("--target_joint", type=str, default="pinky_proximal_joint", 
                   choices=["index_proximal_joint", "middle_proximal_joint", "pinky_proximal_joint", 
                           "ring_proximal_joint", "thumb_proximal_yaw_joint", "thumb_proximal_pitch_joint"],
                   help="Joint to optimize and track error for.")
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
import random
from isaaclab.actuators.actuator_cfg import ImplicitActuatorCfg
from isaaclab.managers import SceneEntityCfg
import isaaclab.utils.math as math_utils

# Import ROS2 components
try:
    import rclpy
    from rclpy.node import Node
    from std_msgs.msg import Float64MultiArray
    ROS2_AVAILABLE = True
except ImportError:
    print("[WARNING]: ROS2 not available. Joint control will not work.")
    ROS2_AVAILABLE = False

##
# Pre-defined configs
##
from isaaclab_assets.robots.inspire_hand import INSPIRE_HAND_CFG

# Genetic Algorithm Configuration - will be updated with command line argument
GENETIC_ALGORITHM_CONFIG = {
    "enabled": True,
    "target_joint": args_cli.target_joint,  # Joint to optimize (from command line)
    "evaluation_interval": 10.0,  # Seconds between evaluations
    "top_performers_ratio": 0.3,  # Top 30% keep their parameters
    "middle_performers_ratio": 0.4,  # Middle 40% mutate
    "bottom_performers_ratio": 0.3,  # Bottom 30% get replaced by top performers
    "stiffness_range": (0.0, 50.0),  # Range for stiffness mutation
    "damping_range": (0.0, 2.0),    # Range for damping mutation
    "mutation_noise": 0.1,  # Standard deviation for sampling around top performers
}

# Available joints for optimization
AVAILABLE_JOINTS = [
    "index_proximal_joint",
    "middle_proximal_joint", 
    "pinky_proximal_joint",
    "ring_proximal_joint",
    "thumb_proximal_yaw_joint",
    "thumb_proximal_pitch_joint"
]

# Joint name to index mapping
JOINT_NAME_TO_INDEX = {
    "index_proximal_joint": 0,
    "middle_proximal_joint": 1, 
    "pinky_proximal_joint": 2,
    "ring_proximal_joint": 3,
    "thumb_proximal_yaw_joint": 4,
    "thumb_proximal_pitch_joint": 9
}

# ROS2 topic index to simulation joint index mapping
# Topic indices: 0=little, 1=ring, 2=middle, 3=index, 4=thumb_pitch, 5=thumb_yaw
TOPIC_TO_SIM_JOINT_MAPPING = {
    0: 2,  # little (topic) -> pinky_proximal_joint (sim)
    1: 3,  # ring (topic) -> ring_proximal_joint (sim)
    2: 1,  # middle (topic) -> middle_proximal_joint (sim)
    3: 0,  # index (topic) -> index_proximal_joint (sim)
    4: 9,  # thumb_pitch (topic) -> thumb_proximal_pitch_joint (sim)
    5: 4   # thumb_yaw (topic) -> thumb_proximal_yaw_joint (sim)
}




class ROS2JointController(Node):
    """ROS2 node to receive joint commands from /real_hand_joints topic."""
    
    def __init__(self):
        super().__init__('isaaclab_joint_controller')
        
        # Subscribe to joint state topic
        self.subscription = self.create_subscription(
            Float64MultiArray,
            '/real_hand_joints',
            self.joint_state_callback,
            10
        )
        
        # Initialize joint values
        self.joint_positions = torch.zeros(12, dtype=torch.float32)
        self.joint_velocities = torch.zeros(12, dtype=torch.float32)
        self.joint_efforts = torch.zeros(12, dtype=torch.float32)
        self.latest_message_received = False
        self.message_count = 0  # Track number of messages received
        
        self.get_logger().info('ROS2 Joint Controller initialized, listening to /real_hand_joints')
    
    def joint_state_callback(self, msg):
        """Callback for joint state messages.
        
        Expects 6 joint angles in the range 0-1000:
        [little, ring, middle, index, thumb_pitch, thumb_yaw]
        """
        try:
            if len(msg.data) != 6:
                self.get_logger().warn(f'Expected 6 joint angles, got {len(msg.data)}')
                return
            
            # Extract the 6 joint angles (0-1000 range)
            little_angle = msg.data[0]      # little finger (pinky)
            ring_angle = msg.data[1]        # ring finger
            middle_angle = msg.data[2]      # middle finger
            index_angle = msg.data[3]       # index finger
            thumb_pitch_angle = msg.data[4] # thumb pitch
            thumb_yaw_angle = msg.data[5]   # thumb yaw
            
            # Reset all joint positions to 0 first
            self.joint_positions.fill_(0.0)
            
            # Map topic indices to simulation joint indices using the correct mapping
            # Most joints: 0-1000 -> 0-1.5
            # Thumb yaw: 0-1000 -> 0.5-1.4
            # Thumb pitch: 0-1000 -> 0-0.9
            
            # Map each topic index to the correct simulation joint index
            topic_data = [
                (0, little_angle),      # little -> pinky_proximal_joint
                (1, ring_angle),        # ring -> ring_proximal_joint
                (2, middle_angle),      # middle -> middle_proximal_joint
                (3, index_angle),       # index -> index_proximal_joint
                (4, thumb_pitch_angle), # thumb_pitch -> thumb_proximal_pitch_joint
                (5, thumb_yaw_angle)    # thumb_yaw -> thumb_proximal_yaw_joint
            ]
            
            for topic_idx, angle in topic_data:
                sim_joint_idx = TOPIC_TO_SIM_JOINT_MAPPING[topic_idx]
                
                # Invert the angle: 0 -> 1000, 1000 -> 0
                inverted_angle = 1000.0 - angle
                
                # Scale based on joint type
                if sim_joint_idx == 8:  # thumb_proximal_yaw_joint
                    scaled_angle = 0.5 + (inverted_angle / 1000.0) * 0.9  # 0.5-1.4
                elif sim_joint_idx == 9:  # thumb_proximal_pitch_joint
                    scaled_angle = (inverted_angle / 1000.0) * 0.9  # 0-0.9
                else:
                    scaled_angle = (inverted_angle / 1000.0) * 1.5  # 0-1.5
                
                self.joint_positions[sim_joint_idx] = scaled_angle
            
            # Set velocities and efforts to 0 (not used)
            self.joint_velocities.fill_(0.0)
            self.joint_efforts.fill_(0.0)
            
            self.latest_message_received = True
            self.message_count += 1
            
            # Log the received values for debugging
            self.get_logger().debug(f'Received joint angles: {msg.data}')
            self.get_logger().debug(f'Mapped positions: {self.joint_positions.tolist()}')
            
        except Exception as e:
            self.get_logger().error(f'Error processing joint state message: {e}')
    
    def get_joint_positions(self):
        """Get current joint positions."""
        return self.joint_positions.clone()
    
    def get_joint_velocities(self):
        """Get current joint velocities."""
        return self.joint_velocities.clone()
    
    def get_joint_efforts(self):
        """Get current joint efforts."""
        return self.joint_efforts.clone()
    
    def has_received_message(self):
        """Check if we've received at least one message."""
        return self.latest_message_received
    
    def get_commanded_positions(self):
        """Get the commanded positions from ROS2 topic."""
        return self.joint_positions.clone()
    
    def get_message_count(self):
        """Get the current message count."""
        return self.message_count

@configclass
class InspireSceneCfg(InteractiveSceneCfg):
    """Configuration for a cart-pole scene."""

    # ground plane
    ground = AssetBaseCfg(prim_path="/World/defaultGroundPlane", spawn=sim_utils.GroundPlaneCfg())

    # lights
    dome_light = AssetBaseCfg(
        prim_path="/World/Light", spawn=sim_utils.DomeLightCfg(intensity=3000.0, color=(0.75, 0.75, 0.75))
    )

    # Create 20 robot configurations with different positions
    # Robots are arranged in a 5x4 grid (5 columns, 4 rows) with 1-meter spacing
    def __post_init__(self):
        super().__post_init__()
        
        # Generate robot configurations programmatically
        num_robots = 20
        cols = 5 # 10 columns
        rows = 4  # 5 rows
        
        for i in range(num_robots):
            robot_num = i + 1
            col = i % cols
            row = i // cols
            
            # Calculate position in grid
            x_pos = col * 1.0
            y_pos = row * 1.0
            
            # Create robot configuration
            robot_cfg = replace(
                INSPIRE_HAND_CFG,
                prim_path=f"{{ENV_REGEX_NS}}/Robot{robot_num}",
                init_state=ArticulationCfg.InitialStateCfg(
                    pos=(x_pos, y_pos, 0.5),
                    rot=(0.0, 0.7071, 0.0, 0.7071),
                    joint_pos={".*": 0.0},
                ),
            )
            
            # Add to the scene configuration
            setattr(self, f"inspire{robot_num}", robot_cfg)

class GeneticAlgorithm:
    """Genetic algorithm for optimizing joint parameters."""
    
    def __init__(self, config, num_robots):
        self.config = config
        self.num_robots = num_robots
        self.target_joint = config["target_joint"]
        self.target_joint_index = JOINT_NAME_TO_INDEX[config["target_joint"]]
        
        # Calculate population sizes
        self.top_size = int(num_robots * config["top_performers_ratio"])
        self.middle_size = int(num_robots * config["middle_performers_ratio"])
        self.bottom_size = num_robots - self.top_size - self.middle_size
        
        # Store robot performance history - track all errors over evaluation period
        self.robot_error_history = [[] for _ in range(num_robots)]  # List of error lists for each robot
        self.robot_errors = torch.zeros(num_robots, dtype=torch.float32)  # Current averaged errors
        self.robot_parameters = []  # List of (stiffness, damping) tuples for each robot
        
        # Initialize random parameters for all robots
        self._initialize_random_parameters()
        
        print(f"[GA]: Initialized genetic algorithm for {self.target_joint}")
        print(f"[GA]: Population sizes - Top: {self.top_size}, Middle: {self.middle_size}, Bottom: {self.bottom_size}")
        print(f"[GA]: Will average errors over {config['evaluation_interval']} second evaluation period")
    
    def _initialize_random_parameters(self):
        """Initialize random stiffness and damping parameters for all robots."""
        for i in range(self.num_robots):
            stiffness = random.uniform(*self.config["stiffness_range"])
            damping = random.uniform(*self.config["damping_range"])
            self.robot_parameters.append((stiffness, damping))
    
    def update_robot_error(self, robot_idx, error):
        """Update the error for a specific robot by adding to history."""
        self.robot_error_history[robot_idx].append(error)
    
    def get_robot_parameters(self, robot_idx):
        """Get the current parameters for a specific robot."""
        return self.robot_parameters[robot_idx]
    
    def evolve_population(self):
        """Evolve the population based on performance."""
        print(f"\n[GA]: Evolving population for {self.target_joint}")
        
        # Calculate average errors from history for each robot
        for robot_idx in range(self.num_robots):
            if self.robot_error_history[robot_idx]:
                avg_error = sum(self.robot_error_history[robot_idx]) / len(self.robot_error_history[robot_idx])
                self.robot_errors[robot_idx] = avg_error
                print(f"[GA]: Robot {robot_idx} - {len(self.robot_error_history[robot_idx])} measurements, avg error: {avg_error:.6f}")
            else:
                self.robot_errors[robot_idx] = float('inf')  # No measurements, consider worst
                print(f"[GA]: Robot {robot_idx} - No measurements, setting error to infinity")
        
        # Sort robots by performance (lower error = better)
        sorted_indices = torch.argsort(self.robot_errors)
        
        # Get top performers
        top_indices = sorted_indices[:self.top_size]
        print(f"[GA]: Top performers (avg errors): {self.robot_errors[top_indices].tolist()}")
        
        # Calculate average parameters of top performers
        top_stiffness = torch.tensor([self.robot_parameters[i][0] for i in top_indices])
        top_damping = torch.tensor([self.robot_parameters[i][1] for i in top_indices])
        avg_stiffness = top_stiffness.mean().item()
        avg_damping = top_damping.mean().item()
        
        print(f"[GA]: Average top performer parameters - Stiffness: {avg_stiffness:.3f}, Damping: {avg_damping:.3f}")
        
        # Keep top performers unchanged
        print(f"[GA]: Keeping top {self.top_size} performers unchanged")
        
        # Mutate middle performers
        middle_indices = sorted_indices[self.top_size:self.top_size + self.middle_size]
        print(f"[GA]: Mutating middle {self.middle_size} performers")
        for idx in middle_indices:
            stiffness = random.uniform(*self.config["stiffness_range"])
            damping = random.uniform(*self.config["damping_range"])
            self.robot_parameters[idx] = (stiffness, damping)
            print(f"[GA]: Robot {idx} -> Stiffness: {stiffness:.3f}, Damping: {damping:.3f}")
        
        # Replace bottom performers with samples around top performers
        bottom_indices = sorted_indices[self.top_size + self.middle_size:]
        print(f"[GA]: Replacing bottom {self.bottom_size} performers")
        for idx in bottom_indices:
            # Sample around the average of top performers with some noise
            stiffness = avg_stiffness + random.gauss(0, self.config["mutation_noise"] * avg_stiffness)
            damping = avg_damping + random.gauss(0, self.config["mutation_noise"] * avg_damping)
            
            # Clamp to valid ranges
            stiffness = max(self.config["stiffness_range"][0], 
                          min(self.config["stiffness_range"][1], stiffness))
            damping = max(self.config["damping_range"][0], 
                         min(self.config["damping_range"][1], damping))
            
            self.robot_parameters[idx] = (stiffness, damping)
            print(f"[GA]: Robot {idx} -> Stiffness: {stiffness:.3f}, Damping: {damping:.3f}")
        
        # Reset error history for next evaluation period
        for robot_idx in range(self.num_robots):
            self.robot_error_history[robot_idx].clear()
        self.robot_errors.fill_(0.0)
        
        print(f"[GA]: Population evolution complete - Error history reset for next evaluation")
    
    def get_best_performance(self):
        """Get the best (lowest) error achieved."""
        return self.robot_errors.min().item()
    
    def get_average_performance(self):
        """Get the average error across all robots."""
        return self.robot_errors.mean().item()
    
    def get_error_statistics(self):
        """Get statistics about error measurements for each robot."""
        stats = {}
        for robot_idx in range(self.num_robots):
            if self.robot_error_history[robot_idx]:
                errors = self.robot_error_history[robot_idx]
                stats[robot_idx] = {
                    'count': len(errors),
                    'min': min(errors),
                    'max': max(errors),
                    'avg': sum(errors) / len(errors),
                    'latest': errors[-1] if errors else 0.0
                }
            else:
                stats[robot_idx] = {
                    'count': 0,
                    'min': 0.0,
                    'max': 0.0,
                    'avg': 0.0,
                    'latest': 0.0
                }
        return stats


def run_simulator(sim: sim_utils.SimulationContext, scene: InteractiveScene):
    """Runs the simulation loop."""
    # Extract scene entities - all 20 robots
    robots = []
    for i in range(1, 21):
        robot_name = f"inspire{i}"
        robots.append(scene[robot_name])
    
    # Initialize genetic algorithm if enabled
    ga = None
    if GENETIC_ALGORITHM_CONFIG["enabled"]:
        ga = GeneticAlgorithm(GENETIC_ALGORITHM_CONFIG, len(robots))
        print(f"[INFO]: Genetic algorithm enabled for joint: {GENETIC_ALGORITHM_CONFIG['target_joint']}")
    
    # Initialize ROS2 if available
    ros2_controller = None
    if ROS2_AVAILABLE:
        try:
            rclpy.init()
            ros2_controller = ROS2JointController()
            print("[INFO]: ROS2 initialized, listening to /real_hand_joints topic")
        except Exception as e:
            print(f"[WARNING]: Failed to initialize ROS2: {e}")
            ros2_controller = None
    
    # Define simulation stepping
    sim_dt = sim.get_physics_dt()
    count = 0
    
    # Track ROS2 message count for error reporting
    last_ros2_message_count = 0
    
    # Timer for genetic algorithm evolution (every 10 seconds by default)
    ga_interval = int(GENETIC_ALGORITHM_CONFIG["evaluation_interval"] / sim_dt)
    last_ga_step = 0
    
    if ga:
        print(f"[INFO]: Genetic algorithm evolution will occur every {ga_interval} steps ({GENETIC_ALGORITHM_CONFIG['evaluation_interval']} seconds)")
    print(f"[INFO]: Managing {len(robots)} robots")
    print(f"[INFO]: Error reports will be printed whenever new ROS2 joint commands are received")
    
    # Simulation loop
    while simulation_app.is_running():
        # Get current targets from ROS2
        if ros2_controller and ros2_controller.has_received_message():
            # Use ROS2 joint commands
            pos_tensor = ros2_controller.get_joint_positions()
            vel_tensor = ros2_controller.get_joint_velocities()
            effort_tensor = ros2_controller.get_joint_efforts()
        else:
            # Use genetic algorithm
            pos_tensor = torch.zeros(12, dtype=torch.float32)
            vel_tensor = torch.zeros(12, dtype=torch.float32)
            effort_tensor = torch.zeros(12, dtype=torch.float32)
        
        # Apply targets to all robots
        for robot in robots:
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
        
        # Process ROS2 callbacks if available
        if ros2_controller:
            try:
                rclpy.spin_once(ros2_controller, timeout_sec=0.0)
            except Exception as e:
                print(f"[WARNING]: ROS2 spin error: {e}")
        
        # Report joint errors every time new ROS2 data is received
        if ros2_controller and ros2_controller.has_received_message():
            current_message_count = ros2_controller.get_message_count()
            if current_message_count > last_ros2_message_count:
                print(f"\n[INFO]: Target Joint Error Report at step {count} (time: {count * sim_dt:.1f}s)")
                print(f"[INFO]: Target Joint: {GENETIC_ALGORITHM_CONFIG['target_joint']}")
                print(f"[INFO]: ROS2 Message #{current_message_count}")
                print("=" * 80)
                
                # Get commanded positions from ROS2
                commanded_positions = ros2_controller.get_commanded_positions()
                
                # Get target joint index
                target_joint_name = GENETIC_ALGORITHM_CONFIG["target_joint"]
                target_joint_idx = JOINT_NAME_TO_INDEX[target_joint_name]
                
                # Report errors for each robot (only target joint)
                robot_errors = []
                for robot_idx, robot in enumerate(robots):
                    # Get actual joint positions from the robot
                    actual_positions = robot.data.joint_pos[0]  # First environment
                    
                    # Get current actuator parameters
                    actuator = robot.actuators["fingers"]
                    current_stiffness = actuator.stiffness[0]  # First environment
                    current_damping = actuator.damping[0]      # First environment
                    
                    # Calculate error for target joint only
                    commanded = commanded_positions[target_joint_idx].item()
                    actual = actual_positions[target_joint_idx].item()
                    target_joint_error = abs(commanded - actual)
                    
                    # Get stiffness and damping for target joint
                    stiffness = current_stiffness[target_joint_idx].item()
                    damping = current_damping[target_joint_idx].item()
                    
                    print(f"Robot {robot_idx + 1:2d}: {target_joint_name:25s} - Commanded={commanded:6.3f}, Actual={actual:6.3f}, Error={target_joint_error:6.3f}, Stiffness={stiffness:6.3f}, Damping={damping:6.3f}")
                    
                    # Store error for summary
                    robot_errors.append((robot_idx + 1, target_joint_error, stiffness, damping))
                    
                    # Update genetic algorithm with target joint error
                    if ga:
                        ga.update_robot_error(robot_idx, target_joint_error)
                
                # Print summary of best and worst performers
                if robot_errors:
                    robot_errors.sort(key=lambda x: x[1])  # Sort by error (best first)
                    best_robot = robot_errors[0]
                    worst_robot = robot_errors[-1]
                    
                    print(f"\n[SUMMARY]: Best Robot {best_robot[0]:2d} (Error: {best_robot[1]:6.3f}, Stiffness: {best_robot[2]:6.3f}, Damping: {best_robot[3]:6.3f})")
                    print(f"[SUMMARY]: Worst Robot {worst_robot[0]:2d} (Error: {worst_robot[1]:6.3f}, Stiffness: {worst_robot[2]:6.3f}, Damping: {worst_robot[3]:6.3f})")
                    print(f"[SUMMARY]: Average Error: {sum(e[1] for e in robot_errors) / len(robot_errors):6.3f}")
                
                print("=" * 80)
                
                # Update the last message count
                last_ros2_message_count = current_message_count
        
        # Genetic algorithm evolution
        if ga and count - last_ga_step >= ga_interval:
            print(f"\n[GA]: Genetic Algorithm Evolution at step {count} (time: {count * sim_dt:.1f}s)")
            print(f"[GA]: Best performance: {ga.get_best_performance():.6f}")
            print(f"[GA]: Average performance: {ga.get_average_performance():.6f}")
            
            # Evolve the population
            ga.evolve_population()
            # Apply new parameters to robots
            print(f"[GA]: Applying new parameters to robots...")
            for robot_idx, robot in enumerate(robots):
                stiffness, damping = ga.get_robot_parameters(robot_idx)
                
                # Apply to the target joint only
                actuator = robot.actuators["fingers"]
                target_joint_idx = ga.target_joint_index
                
                # Update stiffness and damping for the target joint
                new_stiffness = actuator.stiffness.clone()
                new_damping = actuator.damping.clone()
                
                new_stiffness[0, target_joint_idx] = stiffness
                new_damping[0, target_joint_idx] = damping
                
                actuator.stiffness = new_stiffness
                actuator.damping = new_damping
                
                # Write to simulation
                if hasattr(robot, 'write_joint_stiffness_to_sim'):
                    robot.write_joint_stiffness_to_sim(new_stiffness, env_ids=torch.tensor([0]))
                if hasattr(robot, 'write_joint_damping_to_sim'):
                    robot.write_joint_damping_to_sim(new_damping, env_ids=torch.tensor([0]))
            
            print(f"[GA]: Parameters applied successfully")
            last_ga_step = count
    
    # Cleanup
    if ros2_controller:
        try:
            ros2_controller.destroy_node()
            rclpy.shutdown()
        except Exception as e:
            print(f"[WARNING]: Error shutting down ROS2: {e}")


def main():
    """Main function."""
    # Load kit helper
    sim_cfg = sim_utils.SimulationCfg(device=args_cli.device)
    sim = SimulationContext(sim_cfg)
    # Set main camera
    sim.set_camera_view((2.5, 2.5, 4.0), (0.0, 0.0, 2.0))
    # Design scene - use 20 environments for 20 robots
    scene_cfg = InspireSceneCfg(num_envs=1, env_spacing=2.0)
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
