# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""
Simple test script to verify the Inspire Hand configuration without Isaac Sim.
This script only tests the configuration structure, not the actual simulation.
"""

def test_config_import():
    """Test that the configuration can be imported and created."""
    try:
        from inspire_hand_env_cfg import InspireHandEnvCfg
        
        # Create the configuration
        cfg = InspireHandEnvCfg()
        
        print("✓ Successfully imported InspireHandEnvCfg")
        print(f"✓ Configuration created with {cfg.scene.num_envs} environments")
        print(f"✓ Robot has {len(cfg.actions.joint_pos.joint_names)} actuated joints")
        print(f"✓ Observation space: {cfg.observations.policy.concatenate_terms}")
        
        # Test reward terms
        print(f"✓ Reward terms: {len(cfg.rewards.__dict__)} terms")
        for name, term in cfg.rewards.__dict__.items():
            if hasattr(term, 'func'):
                print(f"  - {name}: {term.func.__name__}")
        
        # Test termination terms
        print(f"✓ Termination terms: {len(cfg.terminations.__dict__)} terms")
        for name, term in cfg.terminations.__dict__.items():
            if hasattr(term, 'func'):
                print(f"  - {name}: {term.func.__name__}")
        
        print("\n✓ Configuration test passed!")
        return True
        
    except ImportError as e:
        print(f"✗ Import error: {e}")
        print("This is expected when running outside Isaac Sim environment.")
        return False
    except Exception as e:
        print(f"✗ Configuration error: {e}")
        return False


def test_mdp_functions():
    """Test that the custom MDP functions can be imported."""
    try:
        from mdp.rewards import object_goal_distance, object_goal_orientation, is_success
        from mdp.terminations import object_height_below_threshold
        
        print("✓ Successfully imported custom MDP functions:")
        print("  - object_goal_distance")
        print("  - object_goal_orientation") 
        print("  - is_success")
        print("  - object_height_below_threshold")
        
        return True
        
    except ImportError as e:
        print(f"✗ MDP import error: {e}")
        return False


if __name__ == "__main__":
    print("Testing Inspire Hand Configuration...")
    print("=" * 50)
    
    # Test MDP functions first
    mdp_ok = test_mdp_functions()
    
    # Test configuration
    config_ok = test_config_import()
    
    print("\n" + "=" * 50)
    if mdp_ok and config_ok:
        print("✓ All tests passed! Configuration is ready for Isaac Sim.")
    else:
        print("✗ Some tests failed. Check the errors above.") 