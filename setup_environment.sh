#!/bin/bash

# This script sets up the conda environment for the project.

# Detect the OS
OS="$(uname)"
if [ "$OS" == "Darwin" ]; then
  echo "Setting up environment for macOS..."
  conda env create -f requirements-osx.yml
else
  echo "Setting up environment for Unix/Windows..."
  conda env create -f requirements.yml
fi

# Activate the newly created environment
echo "Activating the 'gam-compare' environment."
conda init
conda activate gam-compare

# Install pip dependencies separately
echo "Installing Pip dependencies..."
pip install -r requirements-pip.txt

echo "Environment setup is complete."
