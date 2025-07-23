![Isaac Lab](docs/source/_static/isaaclab.jpg)

---

# Isaac Lab

[![IsaacSim](https://img.shields.io/badge/IsaacSim-4.5.0-silver.svg)](https://docs.isaacsim.omniverse.nvidia.com/latest/index.html)
[![Python](https://img.shields.io/badge/python-3.10-blue.svg)](https://docs.python.org/3/whatsnew/3.10.html)
[![Linux platform](https://img.shields.io/badge/platform-linux--64-orange.svg)](https://releases.ubuntu.com/20.04/)
[![Windows platform](https://img.shields.io/badge/platform-windows--64-orange.svg)](https://www.microsoft.com/en-us/)
[![pre-commit](https://img.shields.io/github/actions/workflow/status/isaac-sim/IsaacLab/pre-commit.yaml?logo=pre-commit&logoColor=white&label=pre-commit&color=brightgreen)](https://github.com/isaac-sim/IsaacLab/actions/workflows/pre-commit.yaml)
[![docs status](https://img.shields.io/github/actions/workflow/status/isaac-sim/IsaacLab/docs.yaml?label=docs&color=brightgreen)](https://github.com/isaac-sim/IsaacLab/actions/workflows/docs.yaml)
[![License](https://img.shields.io/badge/license-BSD--3-yellow.svg)](https://opensource.org/licenses/BSD-3-Clause)
[![License](https://img.shields.io/badge/license-Apache--2.0-yellow.svg)](https://opensource.org/license/apache-2-0)


**Isaac Lab** is a GPU-accelerated, open-source framework designed to unify and simplify robotics research workflows, such as reinforcement learning, imitation learning, and motion planning. Built on [NVIDIA Isaac Sim](https://docs.isaacsim.omniverse.nvidia.com/latest/index.html), it combines fast and accurate physics and sensor simulation, making it an ideal choice for sim-to-real transfer in robotics.

Isaac Lab provides developers with a range of essential features for accurate sensor simulation, such as RTX-based cameras, LIDAR, or contact sensors. The framework's GPU acceleration enables users to run complex simulations and computations faster, which is key for iterative processes like reinforcement learning and data-intensive tasks. Moreover, Isaac Lab can run locally or be distributed across the cloud, offering flexibility for large-scale deployments.

## Key Features

Isaac Lab offers a comprehensive set of tools and environments designed to facilitate robot learning:
- **Robots**: A diverse collection of robots, from manipulators, quadrupeds, to humanoids, with 16 commonly available models.
- **Environments**: Ready-to-train implementations of more than 30 environments, which can be trained with popular reinforcement learning frameworks such as RSL RL, SKRL, RL Games, or Stable Baselines. We also support multi-agent reinforcement learning.
- **Physics**: Rigid bodies, articulated systems, deformable objects
- **Sensors**: RGB/depth/segmentation cameras, camera annotations, IMU, contact sensors, ray casters.


## Getting Started

Our [documentation page](https://isaac-sim.github.io/IsaacLab) provides everything you need to get started, including detailed tutorials and step-by-step guides. Follow these links to learn more about:

- [Installation steps](https://isaac-sim.github.io/IsaacLab/main/source/setup/installation/index.html#local-installation)
- [Reinforcement learning](https://isaac-sim.github.io/IsaacLab/main/source/overview/reinforcement-learning/rl_existing_scripts.html)
- [Tutorials](https://isaac-sim.github.io/IsaacLab/main/source/tutorials/index.html)
- [Available environments](https://isaac-sim.github.io/IsaacLab/main/source/overview/environments.html)

## Inspire Hand Genetic Algorithm Optimization

This repository includes an advanced genetic algorithm implementation for optimizing Inspire Hand joint parameters with ROS2 integration. The system can optimize stiffness and damping parameters for individual joints while tracking real-time joint commands from ROS2 topics.

### Key Features

- **Genetic Algorithm Optimization**: Automatically optimizes joint stiffness and damping parameters
- **ROS2 Integration**: Receives real-time joint commands from `/real_hand_joints` topic
- **Multi-Robot Testing**: Tests 20 different robot configurations simultaneously
- **Target Joint Selection**: Focus optimization on specific joints (index, middle, pinky, ring, thumb)
- **Real-time Error Tracking**: Monitors and reports joint tracking errors

### Prerequisites

1. **Isaac Sim 4.5.0** installed and configured
2. **ROS2** (for real-time joint control)
3. **Python 3.10** with required dependencies

### Quick Start

#### 1. Basic Usage (Without ROS2)

Run the genetic algorithm optimization with default settings:

```bash
# Navigate to IsaacLab root directory
cd /path/to/IsaacLab

# Run with default target joint (pinky_proximal_joint)
./isaaclab.sh -p scripts/tutorials/02_scene/inspire_scene_multipleconfig.py --num_envs 1

# Run with specific target joint
./isaaclab.sh -p scripts/tutorials/02_scene/inspire_scene_multipleconfig.py --num_envs 1 --target_joint index_proximal_joint
```

#### 2. With ROS2 Integration

1. **Start ROS2 joint publisher** (in a separate terminal):
```bash

Publish to /real_hand_joints topic with your own data
Current expected format: 6 joint angles in range 0-1000
[little, ring, middle, index, thumb_pitch, thumb_yaw]
```

2. **Run the genetic algorithm**:
```bash
./isaaclab.sh -p scripts/tutorials/02_scene/inspire_scene_multipleconfig.py --num_envs 1 --target_joint index_proximal_joint
```

### Available Target Joints

The genetic algorithm can optimize any of these joints:
- `index_proximal_joint` - Index finger proximal joint
- `middle_proximal_joint` - Middle finger proximal joint  
- `pinky_proximal_joint` - Pinky finger proximal joint
- `ring_proximal_joint` - Ring finger proximal joint
- `thumb_proximal_yaw_joint` - Thumb yaw joint
- `thumb_proximal_pitch_joint` - Thumb pitch joint

### Configuration

The genetic algorithm parameters can be modified in `scripts/tutorials/02_scene/inspire_scene_multipleconfig.py`:

```python
GENETIC_ALGORITHM_CONFIG = {
    "enabled": True,
    "target_joint": "index_proximal_joint",
    "evaluation_interval": 10.0,  # Seconds between evaluations
    "top_performers_ratio": 0.3,  # Top 30% keep their parameters
    "middle_performers_ratio": 0.4,  # Middle 40% mutate
    "bottom_performers_ratio": 0.3,  # Bottom 30% get replaced
    "stiffness_range": (0.0, 50.0),  # Range for stiffness mutation
    "damping_range": (0.0, 2.0),    # Range for damping mutation
    "mutation_noise": 0.1,  # Standard deviation for sampling
}
```

### File Structure

```
scripts/tutorials/02_scene/
├── inspire_scene_multipleconfig.py    # Main genetic algorithm script
├── fake_joint_publisher              # ROS2 joint publisher for testing
├── unified_hand_controller.py        # Unified hand controller
├── inspire_scene.py                  # Basic Inspire Hand scene
└── IsaacSim-ros_workspaces/          # ROS2 workspace setup

source/isaaclab_assets/isaaclab_assets/robots/
├── inspire_hand.py                   # Inspire Hand asset configuration
└── inspire_hand_right.usd            # Inspire Hand USD model

source/isaaclab_tasks/isaaclab_tasks/manager_based/inspire_hand/
├── inspire_hand_env_cfg.py           # Environment configuration
├── inspire_hand_env.py               # Environment implementation
├── test_config.py                    # Configuration testing
└── mdp/                              # Custom MDP functions
    ├── rewards.py                    # Reward functions
    └── terminations.py               # Termination conditions
```

### Understanding the Output

The system provides real-time feedback:

1. **Joint Error Reports**: Shows commanded vs actual joint positions for the target joint
2. **Genetic Algorithm Evolution**: Reports population evolution every 10 seconds
3. **Performance Statistics**: Best, worst, and average error across all robots
4. **Parameter Updates**: Shows new stiffness/damping values for each robot

Example output:
```
[INFO]: Target Joint Error Report at step 1500 (time: 15.0s)
[INFO]: Target Joint: index_proximal_joint
Robot  1: index_proximal_joint    - Commanded=0.750, Actual=0.748, Error=0.002, Stiffness=25.123, Damping=1.234
Robot  2: index_proximal_joint    - Commanded=0.750, Actual=0.745, Error=0.005, Stiffness=30.456, Damping=0.987
...

[GA]: Genetic Algorithm Evolution at step 2000 (time: 20.0s)
[GA]: Best performance: 0.001234
[GA]: Average performance: 0.005678
```



### Using Different Robots

The genetic algorithm framework can be adapted to work with other robots. Here's how to modify the code for different robot types:

#### 1. Robot Asset Configuration

Create or modify the robot configuration in `source/isaaclab_assets/isaaclab_assets/robots/`:

```python
# Example: source/isaaclab_assets/isaaclab_assets/robots/your_robot.py
from isaaclab.assets import ArticulationCfg
from isaaclab.utils import configclass

@configclass
class YOUR_ROBOT_CFG(ArticulationCfg):
    """Configuration for Your Robot."""
    
    # Robot-specific parameters
    num_joints = 6  # Update to match your robot's DOF
    joint_names = ["joint_1", "joint_2", "joint_3", "joint_4", "joint_5", "joint_6"]
    
    # USD file path
    usd_file_path = "path/to/your/robot.usd"
    
    # Initial state
    init_state = InitialStateCfg(
        pos=(0.0, 0.0, 0.5),
        rot=(1.0, 0.0, 0.0, 0.0),
        joint_pos={".*": 0.0},
    )
    
    # Actuator configuration
    actuators = {
        "joints": ImplicitActuatorCfg(
            joint_names_expr=[".*"],
            stiffness_range=(0.0, 50.0),
            damping_range=(0.0, 2.0),
        )
    }
```

#### 2. Update Joint Mappings

Modify the joint mapping constants in `scripts/tutorials/02_scene/inspire_scene_multipleconfig.py`:

```python
# Update these constants for your robot
AVAILABLE_JOINTS = [
    "joint_1",
    "joint_2", 
    "joint_3",
    "joint_4",
    "joint_5",
    "joint_6"
]

JOINT_NAME_TO_INDEX = {
    "joint_1": 0,
    "joint_2": 1, 
    "joint_3": 2,
    "joint_4": 3,
    "joint_5": 4,
    "joint_6": 5
}

# Update ROS2 topic to simulation joint mapping
TOPIC_TO_SIM_JOINT_MAPPING = {
    0: 0,  # topic_joint_1 -> sim_joint_1
    1: 1,  # topic_joint_2 -> sim_joint_2
    2: 2,  # topic_joint_3 -> sim_joint_3
    3: 3,  # topic_joint_4 -> sim_joint_4
    4: 4,  # topic_joint_5 -> sim_joint_5
    5: 5   # topic_joint_6 -> sim_joint_6
}
```

#### 3. Modify Scene Configuration

Update the scene configuration class in the same file:

```python
@configclass
class YourRobotSceneCfg(InteractiveSceneCfg):
    """Configuration for your robot scene."""
    
    # Import your robot configuration
    from isaaclab_assets.robots.your_robot import YOUR_ROBOT_CFG
    
    def __post_init__(self):
        super().__post_init__()
        
        # Generate robot configurations programmatically
        num_robots = 20  # Adjust as needed
        cols = 5
        rows = 4
        
        for i in range(num_robots):
            robot_num = i + 1
            col = i % cols
            row = i // cols
            
            x_pos = col * 1.0
            y_pos = row * 1.0
            
            # Create robot configuration with your robot
            robot_cfg = replace(
                YOUR_ROBOT_CFG,  # Use your robot config
                prim_path=f"{{ENV_REGEX_NS}}/Robot{robot_num}",
                init_state=ArticulationCfg.InitialStateCfg(
                    pos=(x_pos, y_pos, 0.5),
                    rot=(0.0, 0.7071, 0.0, 0.7071),
                    joint_pos={".*": 0.0},
                ),
            )
            
            setattr(self, f"robot{robot_num}", robot_cfg)
```

#### 4. Update ROS2 Message Processing

Modify the `ROS2JointController` class to handle your robot's joint data:

```python
def joint_state_callback(self, msg):
    """Callback for joint state messages."""
    try:
        # Update expected message length for your robot
        if len(msg.data) != 6:  # Change to match your robot's DOF
            self.get_logger().warn(f'Expected 6 joint angles, got {len(msg.data)}')
            return
        
        # Extract joint angles for your robot
        joint_1_angle = msg.data[0]
        joint_2_angle = msg.data[1]
        # ... add all your joints
        
        # Reset all joint positions
        self.joint_positions.fill_(0.0)
        
        # Map topic indices to simulation joint indices
        topic_data = [
            (0, joint_1_angle),
            (1, joint_2_angle),
            # ... add all your joints
        ]
        
        for topic_idx, angle in topic_data:
            sim_joint_idx = TOPIC_TO_SIM_JOINT_MAPPING[topic_idx]
            
            # Scale angles appropriately for your robot
            # Most joints: 0-1000 -> 0-1.5 (adjust ranges as needed)
            scaled_angle = (angle / 1000.0) * 1.5
            
            self.joint_positions[sim_joint_idx] = scaled_angle
        
        self.latest_message_received = True
        self.message_count += 1
        
    except Exception as e:
        self.get_logger().error(f'Error processing joint state message: {e}')
```

#### 5. Update Genetic Algorithm Configuration

Modify the genetic algorithm parameters for your robot:

```python
GENETIC_ALGORITHM_CONFIG = {
    "enabled": True,
    "target_joint": "joint_1",  # Change to your target joint
    "evaluation_interval": 10.0,
    "top_performers_ratio": 0.3,
    "middle_performers_ratio": 0.4,
    "bottom_performers_ratio": 0.3,
    "stiffness_range": (0.0, 50.0),  # Adjust for your robot
    "damping_range": (0.0, 2.0),     # Adjust for your robot
    "mutation_noise": 0.1,
}
```



#### 6. Testing Your Changes

1. **Test robot configuration**:
```bash
cd source/isaaclab_assets/isaaclab_assets/robots
python -c "from your_robot import YOUR_ROBOT_CFG; print('Configuration loaded successfully')"
```

2. **Test scene setup**:
```bash
./isaaclab.sh -p scripts/tutorials/02_scene/your_robot_scene.py --num_envs 1
```

3. **Test ROS2 integration**:
```bash
# Publish test data to your robot's topic
ros2 topic pub /real_hand_joints std_msgs/msg/Float64MultiArray "data: [500, 500, 500, 500, 500, 500]"
```

### Connecting to Real Inspire Hand

The genetic algorithm framework can be connected to a real Inspire Hand for hardware-in-the-loop optimization. This enables real-time parameter optimization using actual robot hardware.

#### Prerequisites

1. **Real Inspire Hand** with network connectivity
2. **Network connection** to the robot hand (default IP: `192.168.137.39`, port: `2333`)
3. **Python 3.6+** with required dependencies
4. **ROS2** for real-time data publishing

#### Hardware Setup

1. **Power on the Inspire Hand** and ensure it's connected to your network
2. **Verify network connectivity**:
   ```bash
   ping 192.168.137.39
   ```


#### Interactive Hand Testing

The `interactive_hand_testing.py` script provides a comprehensive interface for testing and controlling the real Inspire Hand.

##### Starting Interactive Testing

```bash
# Navigate to the IPcontrol directory
cd scripts/tutorials/02_scene/IsaacSim-ros_workspaces/humble_ws/IPcontrol

# Run interactive hand testing
python interactive_hand_testing.py
```

##### Available Commands

| Command | Description |
|---------|-------------|
| `read` | Read current joint positions |
| `move <dof> <position>` | Smooth move single joint |
| `moveall <pos1> <pos2>...` | Smooth move all 6 joints |
| `tune <dof>` | Sweep joint back and forth (0-1000) |
| `tune stop` | Stop joint tuning |
| `freq <frequency>` | Set movement frequency (Hz) |
| `inc <increment>` | Set increment size (degrees) |
| `monitor start/stop` | Start/stop position monitoring |
| `print start/stop` | Start/stop continuous position printing |
| `history` | Show position history |
| `clear` | Clear position history |
| `ros start/stop` | Start/stop ROS publishing |
| `publish` | Publish current positions to ROS once |
| `demo` | Run smooth movement demo |
| `config` | Show current configuration |
| `help` | Show help information |
| `quit` | Exit program |

##### Joint Mapping

| DOF | Finger/Component | Joint Name |
|-----|------------------|------------|
| 0 | Little Finger | pinky_proximal_joint |
| 1 | Ring Finger | ring_proximal_joint |
| 2 | Middle Finger | middle_proximal_joint |
| 3 | Index Finger | index_proximal_joint |
| 4 | Thumb Bend | thumb_proximal_pitch_joint |
| 5 | Thumb Rotate | thumb_proximal_yaw_joint |

##### Position Values

- **Range**: 0-1000 (dimensionless)
- **0**: Fully bent position
- **1000**: Fully extended position
- **-1**: No change (maintain current position)

#### Real-Time ROS2 Integration

The interactive testing script can publish joint positions to ROS2 topics for real-time integration with the genetic algorithm.

##### Starting ROS2 Publishing

```bash
# In the interactive testing interface
> ros start
```

This will publish joint positions to `/real_hand_joints` topic in the format expected by the genetic algorithm.



#### Hardware-in-the-Loop Optimization

To run the genetic algorithm with real hardware:

1. **Start interactive hand testing**:
   ```bash
   cd scripts/tutorials/02_scene/IsaacSim-ros_workspaces/humble_ws/IPcontrol
   python interactive_hand_testing.py
   ```

2. **Enable ROS2 publishing**:
   ```
   > ros start
   ```

3. **Set higher frequency or smoother movement**:
   ```
   > freq 90
   ```

4. **Tune one finger**:
   ```
   > tune 0 (for little finger)
   ```

5. **Run the genetic algorithm** (in another terminal):
   ```bash
   ./isaaclab.sh -p scripts/tutorials/02_scene/inspire_scene_multipleconfig.py --num_envs 1 --target_joint pinky_proximal_joint
   ```

6. **Monitor real-time optimization**:
   - The genetic algorithm will receive real joint positions from the hardware
   - Parameter optimization will be based on actual tracking performance
   - Real-time error reports will show hardware vs simulation differences

#### Configuration Files

The real hand control system consists of several key files:

```
scripts/tutorials/02_scene/IsaacSim-ros_workspaces/humble_ws/IPcontrol/
├── interactive_hand_testing.py    # Main interactive interface
├── hand_testing.py                # Advanced hand controller with smooth movement
├── robot_hand_controller.py       # Low-level UDP communication
├── hand_plotter.py                # Real-time plotting and visualization
├── test_with_plot.py              # Testing with real-time plots
└── README.md                      # Detailed hardware documentation
```

#### Network Configuration

The default network configuration can be modified in the controller:

```python
# In robot_hand_controller.py or hand_testing.py
controller = RobotHandController(
    host="192.168.137.39",  # Robot hand IP address
    port=2333,              # UDP port
    hand_id=1               # Hand ID
)
```


### Recording ROS Bags for Training

ROS bags can be used to record real robot data for training agents or for offline analysis and optimization.

#### Recording Joint Data

##### Using Interactive Testing with ROS2 Publishing

1. **Start interactive hand testing with ROS2 publishing**:
   ```bash
   cd scripts/tutorials/02_scene/IsaacSim-ros_workspaces/humble_ws/IPcontrol
   python interactive_hand_testing.py
   ```

2. **Enable ROS2 publishing**:
   ```
   > ros start
   ```

3. **Record joint data** (in another terminal):
   ```bash
   # Record all joint data
   ros2 bag record /real_hand_joints
   
4. **Perform movements** in the interactive interface:
   ```
   > tune 0

   ```

5. **Stop recording**:
   ```bash
   # Press Ctrl+C to stop recording
   ```


##### From Genetic Algorithm

1. **Start the genetic algorithm**:
   ```bash
   ./isaaclab.sh -p scripts/tutorials/02_scene/inspire_scene_multipleconfig.py --num_envs 1 --target_joint index_proximal_joint
   ```
2.**Play ros bag on loop to train**:
# Play back bag data
```bash
ros2 bag play inspire_hand_joints_20250101_120000/ --loop

```



#### Getting Help

- Check the [Inspire Hand README](source/isaaclab_tasks/isaaclab_tasks/manager_based/inspire_hand/README.md) for detailed environment documentation
- Review the demo scripts in `scripts/tutorials/02_scene/` for working examples
- Test ROS2 connectivity with `test_ros2.py` before running the full system
- Refer to existing robot configurations in `source/isaaclab_assets/isaaclab_assets/robots/` for reference
- Check the [IPcontrol README](scripts/tutorials/02_scene/IsaacSim-ros_workspaces/humble_ws/IPcontrol/README.md) for detailed hardware documentation


## Contributing to Isaac Lab

We wholeheartedly welcome contributions from the community to make this framework mature and useful for everyone.
These may happen as bug reports, feature requests, or code contributions. For details, please check our
[contribution guidelines](https://isaac-sim.github.io/IsaacLab/main/source/refs/contributing.html).

## Show & Tell: Share Your Inspiration

We encourage you to utilize our [Show & Tell](https://github.com/isaac-sim/IsaacLab/discussions/categories/show-and-tell) area in the
`Discussions` section of this repository. This space is designed for you to:

* Share the tutorials you've created
* Showcase your learning content
* Present exciting projects you've developed

By sharing your work, you'll inspire others and contribute to the collective knowledge
of our community. Your contributions can spark new ideas and collaborations, fostering
innovation in robotics and simulation.

## Troubleshooting

Please see the [troubleshooting](https://isaac-sim.github.io/IsaacLab/main/source/refs/troubleshooting.html) section for
common fixes or [submit an issue](https://github.com/isaac-sim/IsaacLab/issues).

For issues related to Isaac Sim, we recommend checking its [documentation](https://docs.omniverse.nvidia.com/app_isaacsim/app_isaacsim/overview.html)
or opening a question on its [forums](https://forums.developer.nvidia.com/c/agx-autonomous-machines/isaac/67).

## Support

* Please use GitHub [Discussions](https://github.com/isaac-sim/IsaacLab/discussions) for discussing ideas, asking questions, and requests for new features.
* Github [Issues](https://github.com/isaac-sim/IsaacLab/issues) should only be used to track executable pieces of work with a definite scope and a clear deliverable. These can be fixing bugs, documentation issues, new features, or general updates.

## Connect with the NVIDIA Omniverse Community

Have a project or resource you'd like to share more widely? We'd love to hear from you! Reach out to the
NVIDIA Omniverse Community team at OmniverseCommunity@nvidia.com to discuss potential opportunities
for broader dissemination of your work.

Join us in building a vibrant, collaborative ecosystem where creativity and technology intersect. Your
contributions can make a significant impact on the Isaac Lab community and beyond!

## License

The Isaac Lab framework is released under [BSD-3 License](LICENSE). The `isaaclab_mimic` extension and its corresponding standalone scripts are released under [Apache 2.0](LICENSE-mimic). The license files of its dependencies and assets are present in the [`docs/licenses`](docs/licenses) directory.

## Acknowledgement

Isaac Lab development initiated from the [Orbit](https://isaac-orbit.github.io/) framework. We would appreciate if you would cite it in academic publications as well:

```
@article{mittal2023orbit,
   author={Mittal, Mayank and Yu, Calvin and Yu, Qinxi and Liu, Jingzhou and Rudin, Nikita and Hoeller, David and Yuan, Jia Lin and Singh, Ritvik and Guo, Yunrong and Mazhar, Hammad and Mandlekar, Ajay and Babich, Buck and State, Gavriel and Hutter, Marco and Garg, Animesh},
   journal={IEEE Robotics and Automation Letters},
   title={Orbit: A Unified Simulation Framework for Interactive Robot Learning Environments},
   year={2023},
   volume={8},
   number={6},
   pages={3740-3747},
   doi={10.1109/LRA.2023.3270034}
}
```
