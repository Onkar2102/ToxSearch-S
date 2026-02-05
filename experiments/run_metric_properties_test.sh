#!/bin/bash
# Script to run metric properties verification with correct environment setup

set -e

# Get the script directory
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"

# Activate virtual environment
cd "$PROJECT_ROOT"
if [ -d "venv" ]; then
    source venv/bin/activate
elif [ -d ".venv" ]; then
    source .venv/bin/activate
else
    echo "ERROR: No virtual environment found. Please create one first."
    exit 1
fi

# Change to experiments directory
cd "$SCRIPT_DIR"

# Run the script with default arguments (100 genomes, default directory)
python verify_metric_properties.py --min-genomes 100 "$@"
