#!/bin/bash

# Navigate to validation directory
cd "$(dirname "$0")"

# Create static directory if it doesn't exist
mkdir -p static
mkdir -p templates

# Install requirements if needed
pip install -r requirements.txt

# Run the Flask app
python app.py