#!/bin/bash

# Startup script for Unified Hand Controller
# This script helps launch both the simulation server and GUI client

echo "=== Unified Hand Controller Startup ==="
echo ""

# Check if we're in the right directory
if [[ ! -f "isaaclab.sh" ]]; then
    echo "Error: Please run this script from the IsaacLab root directory"
    exit 1
fi

echo "Choose an option:"
echo "1. Start simulation server (run in Isaac Lab conda env)"
echo "2. Start GUI client (run in venv with PyQt5)"
echo "3. Start both (opens 2 terminals)"
echo ""
read -p "Enter choice (1-3): " choice

case $choice in
    1)
        echo "Starting simulation server..."
        echo "Make sure you're in the Isaac Lab conda environment (env_isaaclab2)"
        ./isaaclab.sh -p scripts/tutorials/02_scene/sim_server.py --num_envs 1
        ;;
    2)
        echo "Starting GUI client..."
        echo "Make sure you're in a venv with PyQt5 and matplotlib installed"
        echo "If not installed, run: pip install PyQt5 matplotlib"
        
        # Try to find Python in common venv locations
        if [[ -f "venv/bin/python" ]]; then
            echo "Using venv/bin/python"
            venv/bin/python gui_client.py
        elif command -v python &> /dev/null; then
            echo "Using system python (make sure PyQt5 is installed)"
            python gui_client.py
        else
            echo "Error: Python not found. Please activate your venv first."
            exit 1
        fi
        ;;
    3)
        echo "Starting both simulation server and GUI client..."
        echo "Opening simulation server in new terminal..."
        
        # Start simulation server in background terminal
        gnome-terminal --title="Simulation Server" -- bash -c "
            echo 'Starting simulation server...';
            echo 'Make sure Isaac Lab conda environment is activated';
            ./isaaclab.sh -p scripts/tutorials/02_scene/sim_server.py --num_envs 1;
            echo 'Simulation server stopped. Press Enter to close.';
            read
        " &
        
        # Wait a moment for server to start
        sleep 3
        
        echo "Opening GUI client in new terminal..."
        gnome-terminal --title="GUI Client" -- bash -c "
            echo 'Starting GUI client...';
            echo 'Make sure venv with PyQt5 is activated';
            if [[ -f 'venv/bin/python' ]]; then
                venv/bin/python gui_client.py;
            else
                python gui_client.py;
            fi;
            echo 'GUI client stopped. Press Enter to close.';
            read
        " &
        
        echo "Both terminals opened. Check the terminal windows."
        ;;
    *)
        echo "Invalid choice. Exiting."
        exit 1
        ;;
esac 