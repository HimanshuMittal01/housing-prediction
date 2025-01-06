#!/bin/bash

# Define clean function
function clean() {
    echo "Cleaning the project of temporary files..."
    find . -type f -name "*.py[co]" -delete
    find . -type d -name "__pycache__" -delete
    if [ $? -eq 0 ]; then
        echo "Project cleaned successfully."
    else
        echo "Error: Failed to clean the project."
        exit 1
    fi
}

# Check for clean argument
if [ "$1" == "clean" ]; then
    clean
    exit 0
fi

# Initialize a new Python project
echo "Initializing a new Python project..."

# Check if uv is installed
if ! command -v uv &> /dev/null
then
    echo "Error: uv is not installed. Please install it first."
    exit 1
fi

# Check if cookiecutter.python_version_number is provided
if [ -z "3.10" ]; then
    echo "Error: Python version number is not specified."
    exit 1
fi

# Create virtual environment and activate it
echo "Creating virtual environment..."
uv venv --python "3.10"
if [ ! -d ".venv" ]; then
    echo "Error: Virtual environment creation failed."
    exit 1
fi
source .venv/bin/activate
if [ $? -ne 0 ]; then
    echo "Error: Failed to activate the virtual environment."
    exit 1
fi

# Initialize git repository
echo "Initializing git repository..."
git init
if [ $? -ne 0 ]; then
    echo "Error: Failed to initialize git repository."
    deactivate
    exit 1
fi

# Install dependencies
echo "Installing dependencies..."
uv add numpy polars scikit-learn loguru typer ipykernel pre-commit
if [ $? -ne 0 ]; then
    echo "Error: Failed to install dependencies."
    deactivate
    exit 1
fi

# Set up pre-commit
echo "Setting up pre-commit hooks..."
.venv/bin/pre-commit install
if [ $? -ne 0 ]; then
    echo "Error: Failed to set up pre-commit hooks."
    deactivate
    exit 1
fi

echo "Project initialization complete. Run 'source .venv/bin/activate' to activate the virtual environment."
