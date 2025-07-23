#!/usr/bin/env python3

"""Simple script to test ROS 2 functionality."""

import sys

def test_ros2():
    """Test if ROS 2 is working."""
    
    print("Testing ROS 2 installation...")
    
    try:
        import rclpy
        print("✓ rclpy imported successfully")
    except ImportError as e:
        print(f"✗ Failed to import rclpy: {e}")
        return False
    
    try:
        from rclpy.node import Node
        print("✓ Node imported successfully")
    except ImportError as e:
        print(f"✗ Failed to import Node: {e}")
        return False
    
    try:
        from std_msgs.msg import Float64MultiArray
        print("✓ std_msgs imported successfully")
    except ImportError as e:
        print(f"✗ Failed to import std_msgs: {e}")
        return False
    
    try:
        # Test basic ROS 2 functionality
        rclpy.init()
        node = rclpy.create_node('test_node')
        print("✓ ROS 2 node created successfully")
        
        # Create a publisher to test
        pub = node.create_publisher(Float64MultiArray, '/test_topic', 10)
        print("✓ Publisher created successfully")
        
        # Cleanup
        node.destroy_node()
        rclpy.shutdown()
        print("✓ ROS 2 cleanup successful")
        
        return True
        
    except Exception as e:
        print(f"✗ ROS 2 functionality test failed: {e}")
        return False

if __name__ == "__main__":
    print("="*50)
    print("ROS 2 FUNCTIONALITY TEST")
    print("="*50)
    
    # Test outside conda environment
    success = test_ros2()
    
    if success:
        print("\n✓ ROS 2 is working correctly!")
        print("\nYou can now use the ROS 2 scripts:")
        print("1. Simulation: ./isaaclab.sh -p scripts/tutorials/02_scene/inspire_sin_scene_ros.py")
        print("2. Real hand: python3 InspireHand/inspire_hand_sdk/example/sin_movement_ros.py")
    else:
        print("\n✗ ROS 2 is not working properly.")
        print("\nTroubleshooting:")
        print("1. Make sure ROS 2 is sourced: source /opt/ros/humble/setup.bash")
        print("2. Try running outside conda environment")
        print("3. Check ROS 2 installation")
    
    print("="*50) 