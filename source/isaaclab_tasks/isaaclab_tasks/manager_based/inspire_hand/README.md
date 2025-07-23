# Inspire Hand Manager-Based Environment

This package provides a manager-based RL environment for the Inspire Hand robot performing in-hand manipulation tasks.

## Overview

The Inspire Hand environment is designed for training reinforcement learning agents to perform in-hand manipulation tasks. The robot must manipulate a DexCube object to reach a goal position and orientation within a specified tolerance.

## Task Description

- **Robot**: Inspire Hand with 12 actuated joints
- **Object**: DexCube that can be manipulated in-hand
- **Goal**: Move the object to a target position and orientation
- **Success**: Object position and orientation within tolerance of goal
- **Failure**: Object falls below threshold height or episode times out

## Configuration

### Robot Configuration
- **Actuated Joints**: 12 DOF (index, middle, pinky, ring, and thumb fingers)
- **Action Space**: Joint position targets
- **Control**: Position control with moving average smoothing

### Observation Space
- Joint positions and velocities
- Fingertip poses (position and orientation)
- Object position, orientation, and velocities
- Goal position and orientation
- Previous actions

### Reward Structure
- **Distance Reward**: Negative distance to goal (tanh kernel)
- **Orientation Reward**: Orientation alignment with goal
- **Action Penalty**: L2 penalty on actions
- **Success Bonus**: Large bonus for task completion
- **Fall Penalty**: Penalty when object falls

### Termination Conditions
- **Success**: Object reaches goal within tolerance
- **Failure**: Object falls below threshold height
- **Timeout**: Episode length exceeded

## Usage

### Running the Environment

To run the environment with Isaac Sim:

```bash
# Navigate to IsaacLab root directory
cd /path/to/IsaacLab

# Run the demo script
./isaaclab.sh -p scripts/tutorials/03_envs/create_inspire_hand_env.py --num_envs 32
```

### Testing Configuration (Without Isaac Sim)

To test the configuration structure without running Isaac Sim:

```bash
cd source/isaaclab_tasks/isaaclab_tasks/manager_based/inspire_hand
python test_config.py
```

### Using in Your Own Script

```python
from isaaclab.app import AppLauncher
from isaaclab.envs import ManagerBasedRLEnv
from isaaclab_tasks.manager_based.inspire_hand import InspireHandEnvCfg

# Initialize Isaac Sim
app_launcher = AppLauncher()
simulation_app = app_launcher.app

# Create environment
env_cfg = InspireHandEnvCfg()
env_cfg.scene.num_envs = 16
env = ManagerBasedRLEnv(cfg=env_cfg)

# Use the environment
obs, rewards, terminated, truncated, info = env.step(actions)
```

## Customization

### Modifying Reward Weights
Edit the `RewardsCfg` class in `inspire_hand_env_cfg.py`:

```python
@configclass
class RewardsCfg:
    object_goal_dist = RewTerm(
        func=rewards.object_goal_distance,
        weight=-10.0,  # Adjust this weight
        params={"std": 0.1, "object_cfg": SceneEntityCfg("object"), "goal_cfg": SceneEntityCfg("goal_marker")},
    )
```

### Adding New Reward Terms
Create new functions in `mdp/rewards.py` and add them to the configuration:

```python
def custom_reward(env: ManagerBasedRLEnv, **params) -> torch.Tensor:
    # Your reward logic here
    return reward_tensor

# Add to RewardsCfg
custom_reward_term = RewTerm(
    func=custom_reward,
    weight=1.0,
    params={...},
)
```

### Domain Randomization
Modify the `EventCfg` class to add or remove randomization:

```python
@configclass
class EventCfg:
    # Add new randomization
    custom_randomization = EventTerm(
        func=mdp.randomize_rigid_body_mass,
        mode="reset",
        params={
            "asset_cfg": SceneEntityCfg("object"),
            "mass_distribution_params": (0.5, 1.5),
            "operation": "scale",
        },
    )
```

## File Structure

```
inspire_hand/
├── __init__.py                 # Package initialization
├── inspire_hand_env_cfg.py     # Main environment configuration
├── test_config.py             # Configuration test script
├── README.md                  # This file
└── mdp/                       # Custom MDP functions
    ├── __init__.py
    ├── rewards.py             # Custom reward functions
    └── terminations.py        # Custom termination functions
```

## Troubleshooting

### Common Issues

1. **ModuleNotFoundError: No module named 'carb'**
   - This occurs when trying to import IsaacLab modules outside Isaac Sim
   - Solution: Always use `AppLauncher` to initialize Isaac Sim first

2. **Configuration Import Errors**
   - Make sure you're running from the correct directory
   - Check that all dependencies are installed

3. **Simulation Errors**
   - Verify the USD file paths are correct
   - Check that the Inspire Hand configuration is properly set up

### Getting Help

- Check the IsaacLab documentation for general usage
- Review the demo script for working examples
- Test the configuration with `test_config.py` first 