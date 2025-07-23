#!/usr/bin/env python3

# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Unified GUI for controlling both simulation and real Inspire Hand with position plotting.

This script provides a single interface to:
1. Send the same sinusoidal commands to both simulation and real hand
2. Plot real-time joint positions from both hands for comparison
3. Control oscillation parameters through a unified GUI

.. code-block:: bash

    # Usage
    ./isaaclab.sh -p scripts/tutorials/02_scene/unified_hand_controller.py --num_envs 1

"""

import argparse
import math
import time
import sys
import threading
import queue
from dataclasses import replace
from contextlib import suppress
from collections import deque

import torch
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg as FigureCanvas


# Isaac Lab imports
from isaaclab.app import AppLauncher

# Add argparse arguments
parser = argparse.ArgumentParser(description="Unified Inspire Hand Controller")
parser.add_argument("--num_envs", type=int, default=1, help="Number of environments to spawn.")
parser.add_argument("--real_hand_port", type=str, default="/dev/serial/by-id/usb-FTDI_FT232R_USB_UART_BG01FESZ-if00-port0", 
                   help="Serial port for real hand")
parser.add_argument("--disable_real_hand", action="store_true", help="Disable real hand connection")
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()

# Launch omniverse app
app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

# Isaac Lab simulation imports
import isaaclab.sim as sim_utils
from isaaclab.assets import ArticulationCfg, AssetBaseCfg
from isaaclab.scene import InteractiveScene, InteractiveSceneCfg
from isaaclab.sim import SimulationContext
from isaaclab.utils import configclass
from isaaclab_assets.robots.inspire_hand import INSPIRE_HAND_CFG

# PyQt5 and real hand imports
try:
    from PyQt5.QtWidgets import (QApplication, QMainWindow, QPushButton, QVBoxLayout, 
                                QWidget, QLabel, QSlider, QHBoxLayout, QComboBox, 
                                QDoubleSpinBox, QCheckBox, QSplitter)
    from PyQt5.QtCore import Qt, QTimer, QThread, pyqtSignal
    
    # Try to import real hand controller
    try:
        sys.path.append('./InspireHand/inspire_hand_sdk/example/')
        from init_set_inspire_hand_485 import ModbusHandler, baud_rates
        REAL_HAND_AVAILABLE = True
    except ImportError:
        print("[WARNING]: Real hand SDK not found. Running in simulation-only mode.")
        REAL_HAND_AVAILABLE = False
        ModbusHandler = None
        baud_rates = {}
        
except ImportError:
    print("[ERROR]: PyQt5 not available. Please install PyQt5 to use this script.")
    sys.exit(1)

# UI components with error handling for headless mode
try:
    import omni.ui
    import isaacsim.gui.components.ui_utils
    OMNI_UI_AVAILABLE = True
except ImportError:
    OMNI_UI_AVAILABLE = False


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


class RealHandController:
    """Controller for the real Inspire Hand."""
    
    def __init__(self, port, enabled=True):
        self.enabled = enabled and REAL_HAND_AVAILABLE
        self.modbus = None
        self.device_id = None
        self.connected = False
        
        if self.enabled:
            self.device_id, self.baudrate = self.find_online_devices(port)
            if self.device_id is not None and self.baudrate is not None:
                self.modbus = ModbusHandler(port, self.baudrate, self.device_id)
                self.connected = True
                self.setup_initial_settings()
                print(f"[INFO]: Connected to real hand: ID={self.device_id}, baudrate={self.baudrate}")
            else:
                print("[WARNING]: No real hand found. Running in simulation-only mode.")
                self.enabled = False
    
    def find_online_devices(self, port):
        """Find online devices on the serial port"""
        if not REAL_HAND_AVAILABLE:
            return None, None
            
        for baudrate_name, rate_value in baud_rates.items():
            for device_id in range(100):
                try:
                    modbus = ModbusHandler(port, rate_value, device_id)
                    res = modbus.read_register(1000, 1)
                    if res is not None:
                        modbus.close()
                        return device_id, rate_value
                    modbus.close()
                except Exception:
                    pass
        return None, None
    
    def setup_initial_settings(self):
        """Set initial speed and force settings for all fingers"""
        if not self.connected or not self.modbus:
            return
            
        try:
            # Set default speeds for all fingers
            speeds = [500] * 6
            self.modbus.write_registers(1032, speeds)
            
            # Set default forces for all fingers  
            forces = [300] * 6
            self.modbus.write_registers(1044, forces)
            
            print("[INFO]: Initial real hand settings applied")
        except Exception as e:
            print(f"[ERROR]: Error setting initial real hand parameters: {e}")
    
    def set_finger_position(self, finger_index, position):
        """Set position for a specific finger (0-1000 range)"""
        if not self.connected:
            return
            
        try:
            # Convert position to 0-1000 range and clamp
            position = max(0, min(1000, int(position)))
            angles = [500] * 6  # Keep other fingers at center
            angles[finger_index] = position
            self.modbus.write_registers(1486, angles)
        except Exception as e:
            print(f"[ERROR]: Error setting real hand position: {e}")
    
    def get_finger_positions(self):
        """Get current finger positions from real hand"""
        if not self.connected:
            return [0] * 6
            
        try:
            # Read current positions (this might vary based on your hand's registers)
            positions = self.modbus.read_registers(1486, 6)
            return positions if positions else [0] * 6
        except Exception as e:
            print(f"[ERROR]: Error reading real hand positions: {e}")
            return [0] * 6
    
    def emergency_stop(self):
        """Emergency stop - open all fingers"""
        if not self.connected:
            return
            
        try:
            angles = [0] * 6
            self.modbus.write_registers(1486, angles)
            print("[INFO]: Real hand emergency stop executed")
        except Exception as e:
            print(f"[ERROR]: Error in real hand emergency stop: {e}")
    
    def close(self):
        """Close connection to real hand"""
        if self.modbus:
            self.modbus.close()
            self.connected = False


class SimulationThread(QThread):
    """Thread for running Isaac Lab simulation"""
    
    position_updated = pyqtSignal(list)  # Signal to emit joint positions
    
    def __init__(self, scene_cfg):
        super().__init__()
        self.scene_cfg = scene_cfg
        self.sim = None
        self.scene = None
        self.robot = None
        self.running = False
        
        # Shared control parameters
        self.selected_finger = 0
        self.oscillation_speed = 1.0
        self.oscillation_amplitude = 0.5
        self.oscillation_offset = 0.75
        self.start_time = time.time()
        
    def setup_simulation(self):
        """Setup the simulation environment"""
        sim_cfg = sim_utils.SimulationCfg(device=args_cli.device, dt=0.02)  # 50 Hz
        self.sim = SimulationContext(sim_cfg)
        self.sim.set_camera_view((1.5, 1.5, 4.0), (0.0, 0.0, 2.0))
        
        self.scene = InteractiveScene(self.scene_cfg)
        self.robot = self.scene["inspire"]
        
        self.sim.reset()
        print("[INFO]: Simulation setup complete")
    
    def update_parameters(self, finger_index, speed, amplitude, offset):
        """Update oscillation parameters"""
        self.selected_finger = finger_index
        self.oscillation_speed = speed
        self.oscillation_amplitude = amplitude
        self.oscillation_offset = offset
    
    def reset_timer(self):
        """Reset the oscillation timer"""
        self.start_time = time.time()
    
    def get_joint_positions(self):
        """Get current joint position targets with sinusoidal oscillation"""
        current_time = time.time() - self.start_time
        
        # Calculate sinusoidal position for selected finger
        sin_value = math.sin(2 * math.pi * self.oscillation_speed * current_time)
        oscillating_position = self.oscillation_offset + self.oscillation_amplitude * sin_value
        
        # Create joint position array (12 joints)
        joint_values = [0.75] * 12
        if self.selected_finger < len(joint_values):
            joint_values[self.selected_finger] = oscillating_position
            
        return torch.tensor(joint_values, dtype=torch.float32)
    
    def run(self):
        """Main simulation loop"""
        if not self.sim:
            self.setup_simulation()
        
        self.running = True
        sim_dt = self.sim.get_physics_dt()
        
        while self.running and simulation_app.is_running():
            try:
                # Get current targets
                pos_tensor = self.get_joint_positions()
                
                # Apply targets to robot
                self.robot.set_joint_position_target(pos_tensor)
                self.scene.write_data_to_sim()
                
                # Perform step
                self.sim.step()
                self.scene.update(sim_dt)
                
                # Get actual joint positions and emit signal
                actual_positions = self.robot.data.joint_pos[0].cpu().numpy().tolist()
                self.position_updated.emit(actual_positions)
                
                time.sleep(0.01)  # Small delay to prevent excessive CPU usage
                
            except Exception as e:
                print(f"[ERROR]: Simulation error: {e}")
                break
    
    def stop(self):
        """Stop the simulation thread"""
        self.running = False


class UnifiedHandController(QMainWindow):
    """Main GUI controller for both simulation and real hand"""
    
    def __init__(self):
        super().__init__()
        
        # Initialize components
        self.real_hand = RealHandController(args_cli.real_hand_port, not args_cli.disable_real_hand)
        
        # Simulation setup
        scene_cfg = InspireSceneCfg(num_envs=args_cli.num_envs, env_spacing=2.0)
        self.sim_thread = SimulationThread(scene_cfg)
        self.sim_thread.position_updated.connect(self.update_sim_positions)
        
        # Control parameters
        self.oscillation_active = False
        self.selected_finger = 0
        self.frequency = 1.0
        self.amplitude = 0.5
        self.center_position = 0.75
        
        # Data storage for plotting
        self.max_data_points = 500
        self.time_data = deque(maxlen=self.max_data_points)
        self.sim_data = deque(maxlen=self.max_data_points)
        self.real_data = deque(maxlen=self.max_data_points)
        self.start_time = time.time()
        
        # Finger mapping (map to real hand finger indices)
        self.finger_options = {
            "Little Finger": 0,
            "Ring Finger": 1, 
            "Middle Finger": 2,
            "Index Finger": 3,
            "Thumb Bend": 4,
            "Thumb Rotate": 5
        }
        
        # Simulation finger mapping
        self.sim_finger_map = {
            "Little Finger": 2,  # pinky_proximal
            "Ring Finger": 3,    # ring_proximal
            "Middle Finger": 1,  # middle_proximal
            "Index Finger": 0,   # index_proximal
            "Thumb Bend": 9,     # thumb_pitch
            "Thumb Rotate": 4    # thumb_yaw
        }
        
        self.initUI()
        
        # Start update timer
        self.update_timer = QTimer()
        self.update_timer.timeout.connect(self.update_real_hand)
        self.update_timer.setInterval(20)  # 50 Hz
        
    def initUI(self):
        """Initialize the user interface"""
        self.setWindowTitle('Unified Inspire Hand Controller')
        self.setGeometry(100, 100, 1200, 800)
        
        # Create main widget and layout
        main_widget = QWidget()
        main_layout = QHBoxLayout()
        
        # Create splitter for controls and plot
        splitter = QSplitter(Qt.Horizontal)
        
        # Control panel
        control_panel = self.create_control_panel()
        splitter.addWidget(control_panel)
        
        # Plot panel
        plot_panel = self.create_plot_panel()
        splitter.addWidget(plot_panel)
        
        # Set splitter proportions
        splitter.setSizes([300, 900])
        
        main_layout.addWidget(splitter)
        main_widget.setLayout(main_layout)
        self.setCentralWidget(main_widget)
        
        self.show()
    
    def create_control_panel(self):
        """Create the control panel widget"""
        panel = QWidget()
        layout = QVBoxLayout()
        
        # Title
        title = QLabel("Hand Controller")
        title.setStyleSheet("font-size: 16px; font-weight: bold;")
        layout.addWidget(title)
        
        # Connection status
        sim_status = QLabel("Simulation: Ready")
        sim_status.setStyleSheet("color: green;")
        layout.addWidget(sim_status)
        
        real_status = QLabel(f"Real Hand: {'Connected' if self.real_hand.connected else 'Disconnected'}")
        real_status.setStyleSheet(f"color: {'green' if self.real_hand.connected else 'red'};")
        layout.addWidget(real_status)
        
        layout.addWidget(QLabel(""))  # Spacer
        
        # Finger selection
        layout.addWidget(QLabel("Finger:"))
        self.finger_combo = QComboBox()
        self.finger_combo.addItems(list(self.finger_options.keys()))
        self.finger_combo.currentIndexChanged.connect(self.on_finger_changed)
        layout.addWidget(self.finger_combo)
        
        # Frequency control
        layout.addWidget(QLabel("Frequency (Hz):"))
        self.freq_spinbox = QDoubleSpinBox()
        self.freq_spinbox.setRange(0.1, 5.0)
        self.freq_spinbox.setSingleStep(0.1)
        self.freq_spinbox.setValue(1.0)
        self.freq_spinbox.setDecimals(1)
        self.freq_spinbox.valueChanged.connect(self.on_frequency_changed)
        layout.addWidget(self.freq_spinbox)
        
        # Amplitude control
        layout.addWidget(QLabel("Amplitude:"))
        self.amp_slider = QSlider(Qt.Horizontal)
        self.amp_slider.setRange(10, 100)
        self.amp_slider.setValue(50)
        self.amp_slider.valueChanged.connect(self.on_amplitude_changed)
        self.amp_label = QLabel("0.50")
        amp_layout = QHBoxLayout()
        amp_layout.addWidget(self.amp_slider)
        amp_layout.addWidget(self.amp_label)
        amp_widget = QWidget()
        amp_widget.setLayout(amp_layout)
        layout.addWidget(amp_widget)
        
        # Center position control
        layout.addWidget(QLabel("Center Position:"))
        self.center_slider = QSlider(Qt.Horizontal)
        self.center_slider.setRange(0, 150)
        self.center_slider.setValue(75)
        self.center_slider.valueChanged.connect(self.on_center_changed)
        self.center_label = QLabel("0.75")
        center_layout = QHBoxLayout()
        center_layout.addWidget(self.center_slider)
        center_layout.addWidget(self.center_label)
        center_widget = QWidget()
        center_widget.setLayout(center_layout)
        layout.addWidget(center_widget)
        
        layout.addWidget(QLabel(""))  # Spacer
        
        # Control buttons
        self.start_button = QPushButton('Start Oscillation')
        self.start_button.clicked.connect(self.toggle_oscillation)
        layout.addWidget(self.start_button)
        
        reset_button = QPushButton('Reset Timer')
        reset_button.clicked.connect(self.reset_timer)
        layout.addWidget(reset_button)
        
        stop_button = QPushButton('Stop & Reset')
        stop_button.clicked.connect(self.stop_and_reset)
        layout.addWidget(stop_button)
        
        # Emergency stop
        emergency_button = QPushButton('EMERGENCY STOP')
        emergency_button.setStyleSheet("background-color: red; color: white; font-weight: bold;")
        emergency_button.clicked.connect(self.emergency_stop)
        layout.addWidget(emergency_button)
        
        layout.addStretch()  # Push everything to top
        
        panel.setLayout(layout)
        panel.setMaximumWidth(300)
        return panel
    
    def create_plot_panel(self):
        """Create the plotting panel"""
        panel = QWidget()
        layout = QVBoxLayout()
        
        # Create matplotlib figure
        self.fig, self.ax = plt.subplots(figsize=(10, 6))
        self.canvas = FigureCanvas(self.fig)
        layout.addWidget(self.canvas)
        
        # Setup plot
        self.ax.set_title('Joint Position Comparison')
        self.ax.set_xlabel('Time (s)')
        self.ax.set_ylabel('Joint Position')
        self.ax.grid(True, alpha=0.3)
        
        # Initialize empty lines
        self.sim_line, = self.ax.plot([], [], 'b-', label='Simulation', linewidth=2)
        self.real_line, = self.ax.plot([], [], 'r-', label='Real Hand', linewidth=2)
        self.ax.legend()
        
        panel.setLayout(layout)
        return panel
    
    def on_finger_changed(self):
        """Handle finger selection change"""
        finger_name = self.finger_combo.currentText()
        self.selected_finger = self.finger_options[finger_name]
        
        # Update simulation finger index
        sim_finger_idx = self.sim_finger_map.get(finger_name, 0)
        self.update_simulation_parameters(sim_finger_idx)
        
        # Clear previous data when changing fingers
        self.time_data.clear()
        self.sim_data.clear()
        self.real_data.clear()
    
    def on_frequency_changed(self):
        """Handle frequency change"""
        self.frequency = self.freq_spinbox.value()
        finger_name = self.finger_combo.currentText()
        sim_finger_idx = self.sim_finger_map.get(finger_name, 0)
        self.update_simulation_parameters(sim_finger_idx)
    
    def on_amplitude_changed(self):
        """Handle amplitude change"""
        self.amplitude = self.amp_slider.value() / 100.0
        self.amp_label.setText(f"{self.amplitude:.2f}")
        finger_name = self.finger_combo.currentText()
        sim_finger_idx = self.sim_finger_map.get(finger_name, 0)
        self.update_simulation_parameters(sim_finger_idx)
    
    def on_center_changed(self):
        """Handle center position change"""
        self.center_position = self.center_slider.value() / 100.0
        self.center_label.setText(f"{self.center_position:.2f}")
        finger_name = self.finger_combo.currentText()
        sim_finger_idx = self.sim_finger_map.get(finger_name, 0)
        self.update_simulation_parameters(sim_finger_idx)
    
    def update_simulation_parameters(self, sim_finger_idx=None):
        """Update simulation parameters"""
        if sim_finger_idx is None:
            finger_name = self.finger_combo.currentText()
            sim_finger_idx = self.sim_finger_map.get(finger_name, 0)
            
        if hasattr(self.sim_thread, 'update_parameters'):
            self.sim_thread.update_parameters(
                sim_finger_idx, self.frequency, self.amplitude, self.center_position
            )
    
    def toggle_oscillation(self):
        """Toggle oscillation on/off"""
        if not self.oscillation_active:
            self.start_oscillation()
        else:
            self.stop_oscillation()
    
    def start_oscillation(self):
        """Start oscillation"""
        self.oscillation_active = True
        self.start_time = time.time()
        
        # Start simulation thread
        if not self.sim_thread.isRunning():
            self.sim_thread.start()
        
        # Start real hand updates
        self.update_timer.start()
        
        self.start_button.setText('Stop Oscillation')
        print("[INFO]: Started oscillation for both hands")
    
    def stop_oscillation(self):
        """Stop oscillation"""
        self.oscillation_active = False
        self.update_timer.stop()
        self.start_button.setText('Start Oscillation')
        print("[INFO]: Stopped oscillation")
    
    def reset_timer(self):
        """Reset oscillation timer"""
        self.start_time = time.time()
        self.sim_thread.reset_timer()
        # Clear plot data
        self.time_data.clear()
        self.sim_data.clear()
        self.real_data.clear()
    
    def stop_and_reset(self):
        """Stop and reset both hands"""
        self.stop_oscillation()
        # Reset real hand to center position (0.75 in sim range = 500 in real hand range)
        if self.real_hand.connected:
            center_in_real_range = (0.75 / 1.5) * 1000  # Convert 0.75 from sim range to real range
            self.real_hand.set_finger_position(self.selected_finger, int(center_in_real_range))
    
    def emergency_stop(self):
        """Emergency stop for both hands"""
        self.stop_oscillation()
        if self.real_hand.connected:
            self.real_hand.emergency_stop()
        print("[INFO]: Emergency stop executed")
    
    def update_real_hand(self):
        """Update real hand position"""
        if not self.oscillation_active or not self.real_hand.connected:
            return
        
        # Calculate sinusoidal position in simulation range (0-1.5)
        current_time = time.time() - self.start_time
        sin_value = math.sin(2 * math.pi * self.frequency * current_time)
        sim_position = self.center_position + self.amplitude * sin_value
        
        # Convert from simulation range (0-1.5) to real hand range (0-1000)
        # Scale: sim_position / 1.5 * 1000 to maintain proportional mapping
        position = (sim_position / 1.5) * 1000
        position = max(0, min(1000, position))
        
        # Set real hand position
        self.real_hand.set_finger_position(self.selected_finger, position)
        
        # Get actual position for plotting
        actual_positions = self.real_hand.get_finger_positions()
        if actual_positions and len(actual_positions) > self.selected_finger:
            # Convert from real hand range (0-1000) back to simulation range (0-1.5) for plotting
            real_pos = (actual_positions[self.selected_finger] / 1000.0) * 1.5
            
            # Add to plot data
            current_plot_time = time.time() - self.start_time
            self.time_data.append(current_plot_time)
            self.real_data.append(real_pos)
            
            # Update plot
            self.update_plot()
    
    def update_sim_positions(self, positions):
        """Update simulation positions (called from simulation thread)"""
        if not self.oscillation_active:
            return
            
        finger_name = self.finger_combo.currentText()
        sim_finger_idx = self.sim_finger_map.get(finger_name, 0)
        
        if len(positions) <= sim_finger_idx:
            return
        
        sim_pos = positions[sim_finger_idx]
        current_plot_time = time.time() - self.start_time
        
        # Synchronize with real hand data
        if len(self.time_data) == 0 or current_plot_time > self.time_data[-1]:
            self.time_data.append(current_plot_time)
        
        self.sim_data.append(sim_pos)
        
        # Keep data lengths synchronized
        while len(self.sim_data) > len(self.time_data):
            self.sim_data.popleft()
        while len(self.real_data) > len(self.time_data):
            self.real_data.popleft()
    
    def update_plot(self):
        """Update the plot with current data"""
        if len(self.time_data) == 0:
            return
        
        # Convert deques to lists for plotting
        time_list = list(self.time_data)
        sim_list = list(self.sim_data) if self.sim_data else []
        real_list = list(self.real_data) if self.real_data else []
        
        # Update simulation line
        if sim_list:
            sim_time = time_list[:len(sim_list)]
            self.sim_line.set_data(sim_time, sim_list)
        
        # Update real hand line
        if real_list:
            real_time = time_list[:len(real_list)]
            self.real_line.set_data(real_time, real_list)
        
        # Update plot limits
        if time_list:
            self.ax.set_xlim(max(0, time_list[-1] - 10), time_list[-1] + 1)
            
            all_data = sim_list + real_list
            if all_data:
                y_min, y_max = min(all_data), max(all_data)
                margin = (y_max - y_min) * 0.1 if y_max != y_min else 0.1
                self.ax.set_ylim(y_min - margin, y_max + margin)
        
        # Refresh canvas
        self.canvas.draw()
    
    def closeEvent(self, event):
        """Handle application close"""
        self.stop_oscillation()
        
        # Stop simulation thread
        if self.sim_thread.isRunning():
            self.sim_thread.stop()
            self.sim_thread.wait(3000)  # Wait up to 3 seconds
        
        # Close real hand connection
        self.real_hand.close()
        
        event.accept()


def main():
    """Main function"""
    app = QApplication(sys.argv)
    
    try:
        controller = UnifiedHandController()
        
        # Handle Ctrl+C gracefully
        import signal
        def signal_handler(sig, frame):
            print("\n[INFO]: Shutting down...")
            controller.close()
            app.quit()
        
        signal.signal(signal.SIGINT, signal_handler)
        
        sys.exit(app.exec_())
        
    except Exception as e:
        print(f"[ERROR]: Application error: {e}")
        sys.exit(1)
    finally:
        # Close simulation app
        if 'simulation_app' in globals():
            simulation_app.close()


if __name__ == "__main__":
    main() 