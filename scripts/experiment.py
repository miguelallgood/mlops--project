import os
import pandas as pd
import mlflow
import mlflow.sklearn
import optuna
from sklearn.metrics import accuracy_score
from preprocessing import DataPreprocessor
from sklearn.ensemble import RandomForestClassifier

# Set MLflow tracking URI
mlflow.set_tracking_uri("http://localhost:5000")

# Function to check and create or use existing experiment
def get_or_create_experiment(experiment_name):
    try:
        client = mlflow.tracking.MlflowClient()
        experiment_id = client.create_experiment(experiment_name)
        print(f"Created new experiment with name: {experiment_name}")
    except mlflow.exceptions.MlflowException:
        experiment = client.get_experiment_by_name(experiment_name)
        experiment_id = experiment.experiment_id
        print(f"Using existing experiment with name: {experiment_name}")
    return experiment_id

# Ensure no active run is present
if mlflow.active_run():
    mlflow.end_run()

# Create or use existing experiment
experiment_name = "Customer_churning"
experiment_id = get_or_create_experiment(experiment_name)

# Start an MLflow run
try:
    with mlflow.start_run(experiment_id=experiment_id):
        # Load the dataset
        current_directory = os.getcwd()
        csv_file_path = os.path.join(current_directory, "data", "BankChurners.csv")
        churners_df = pd.read_csv(csv_file_path, sep=',')

        # Instantiate the DataPreprocessor class
        preprocessor = DataPreprocessor(df_name=churners_df)

        # Preprocess the data
        X_train, X_test, y_train, y_test = preprocessor.preprocess()

        def objective(trial):
            # Define hyperparameters
            params = {
                "n_estimators": trial.suggest_int("n_estimators", 10, 100),
                "max_depth": trial.suggest_int("max_depth", 1, 32),
                "min_samples_split": trial.suggest_float("min_samples_split", 0.1, 1.0),
                "min_samples_leaf": trial.suggest_float("min_samples_leaf", 0.1, 0.5),
                "max_features": trial.suggest_categorical("max_features", [None, "sqrt", "log2"]),
                "bootstrap": trial.suggest_categorical("bootstrap", [True, False])
            }

            # Train Random Forest Classifier
            rf = RandomForestClassifier(**params)
            rf.fit(X_train, y_train)
            preds = rf.predict(X_test)
            accuracy = accuracy_score(y_test, preds)

            # Log to MLflow
            with mlflow.start_run(nested=True):
                mlflow.log_params(params)
                mlflow.log_metric("accuracy", accuracy)

            return 1 - accuracy

        # Initialize the Optuna study
        study = optuna.create_study(direction="minimize")

        # Execute the hyperparameter optimization trials
        study.optimize(objective, n_trials=50)

        # Log best parameters and accuracy
        mlflow.log_params(study.best_params)
        mlflow.log_metric("best_accuracy", 1 - study.best_value)

        # Train final model with best parameters
        best_params = study.best_params
        model = RandomForestClassifier(**best_params)
        model.fit(X_train, y_train)

        # Log the model and artifacts
        mlflow.sklearn.log_model(model, "rfc")
        # mlflow.log_artifacts("artifacts")

finally:
    if mlflow.active_run():
        mlflow.end_run()
