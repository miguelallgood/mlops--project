#!/bin/bash

# Start the MLflow server in the background
mlflow server --backend-store-uri sqlite:///mlflow.db --default-artifact-root ./mlruns --host 0.0.0.0 --port 5000 &

# Wait for MLflow server to start
sleep 10

# Run the experiment script
python experiment.py

# Keep the container running to access MLflow UI
tail -f /dev/null