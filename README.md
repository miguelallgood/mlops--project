# MLflow Project with Docker and Terraform

This repository contains code and configuration files to set up an MLflow experiment environment using Docker and Terraform for infrastructure management. The experiment uses Optuna for hyperparameter tuning.

## Prerequisites

Before you begin, ensure you have met the following requirements:

- Docker installed on your system.
- Terraform installed on your system.
- Python installed on your system.

## Steps to Set Up MLflow Experiment Environment

1. **Clone the Repository:**
    ```bash
    git clone https://github.com/miguelallgood/mlops--project.git
2. **Navigate to the Root Directory:**
    ```bash
    cd mlops--project
3. **Build the Docker Image:**
    ```bash    
    terraform init
    terraform apply
This command builds the Docker image using the Dockerfile provided and sets up the necessary Docker container with MLflow.

4. **Run the Experiment:**
Once the Docker container is up and running, it executes the MLflow experiment. You can access the MLflow UI at http://localhost:5000 in your web browser.

5. **Model Serving**:
   - After training, the model is saved and served using MLflow's model serving capabilities.
   - The model is served on port 1234, which can be accessed locally.

6. **Terminate the Docker Container:**
After you have completed your experiment, you can terminate the Docker container by running:
    ```bash
    terraform destroy
This command will remove all resources created by Terraform, including the Docker container.

## Project Structure

**scripts:** Contains the dataset and scripts for setting up the environment and running the MLflow experiment.

```
mlops-project
├── Dockerfile
├── main.tf
├── .gitignore
├── scripts
│   ├── experiment.py
│   ├── entrypoint.sh
│   ├── requirements.txt
│   └── data
└── artifacts
```

## Additional Information

- Optuna is used for hyperparameter tuning in the experiment. Optuna is an open-source hyperparameter optimization framework to automate hyperparameter search. The default method used is Tree-structured Parzen Estimator (TPE), a Bayesian optimization algorithm. The results of the tuning process, including the best hyperparameters found, are logged in MLflow. 
