#!/bin/bash

# Create and activate virtual environment
python -m venv venv
source venv/bin/activate

# Install pip-tools for dependency management
pip install pip-tools

# Install dependencies
pip install -r requirements.txt

# Run verification script
python verify_setup.py

echo "Setup complete! Run 'streamlit run streamlit_app.py' to start the application."
