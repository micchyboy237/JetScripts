#!/bin/zsh

# Script to fix Python environment and test discovery issues
# Run in /Users/jethroestrada/Desktop/External_Projects/Jet_Projects

# Exit on error
set -e

# Define project root
PROJECT_ROOT="/Users/jethroestrada/Desktop/External_Projects/Jet_Projects"

# 1. Verify and set up Pixi environment (if using Pixi)
if ! command -v pixi &> /dev/null; then
    echo "Installing Pixi..."
    curl -fsSL https://pixi.sh/install.sh | bash
    # Add Pixi to shell (update ~/.zshrc if needed)
    echo 'eval "$(pixi completion --shell zsh)"' >> ~/.zshrc
    source ~/.zshrc
fi

# Initialize Pixi environment if .pixi/envs does not exist
if [ ! -d "$PROJECT_ROOT/.pixi/envs" ]; then
    echo "Initializing Pixi environment..."
    cd $PROJECT_ROOT
    pixi init
    pixi install python
fi

# 2. Set up Python environment with pyenv and virtualenv
echo "Verifying pyenv and Python 3.12.9..."
pyenv install 3.12.9 -s  # Install if not present, skip if already installed
pyenv local 3.12.9  # Set Python version for project
python --version

# Create and activate virtual environment
echo "Creating virtual environment..."
python -m venv .venv
source .venv/bin/activate

# Install dependencies from requirements-frozen.txt
echo "Installing dependencies..."
pip install --upgrade pip
pip install -r jet_python_modules/requirements-frozen.txt

# 3. Update PYTHONPATH to include JetScripts directory
echo "Updating PYTHONPATH..."
export PYTHONPATH=$PYTHONPATH:$PROJECT_ROOT/jet_python_modules:$PROJECT_ROOT/JetScripts
echo 'export PYTHONPATH=$PYTHONPATH:'"$PROJECT_ROOT/jet_python_modules:$PROJECT_ROOT/JetScripts" >> ~/.zshrc

# 4. Update pytest configuration
echo "Configuring pytest.ini..."
cat > jet_python_modules/.pytest.ini << EOL
[pytest]
python_files = test_*.py
python_functions = test_*
pythonpath = .
timeout = 30
EOL

# 5. Install and update pytest and plugins
echo "Updating pytest and plugins..."
pip install --upgrade pytest pytest-asyncio pytest-playwright

# 6. Clear Python extension cache
echo "Clearing Python extension cache..."
rm -rf ~/.cursor/extensions/ms-python.python-*

# 7. Run pytest manually to verify
echo "Running pytest to verify setup..."
python -m pytest -v JetScripts/benchmark/test_execute_extract_job_entities.py

# 8. Deactivate virtual environment
deactivate

echo "Setup complete. Please restart Cursor and select the virtual environment (.venv/bin/python) in the Python: Select Interpreter command."