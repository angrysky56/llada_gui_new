#!/bin/bash

# Navigate to the application directory
cd "$(dirname "$0")"

# Activate the virtual environment and run the application
./venv/bin/python run.py
