import os
import pickle
import datetime
import argparse
from joblib import dump

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
import mlflow

if __name__ == "__main__":
    # ----------------------------
    # Parse timestamp from GitHub Actions--
    # ----------------------------
    parser = argparse.ArgumentParser()
    parser.add_argument("--timestamp", type=str, required=True, help="Timestamp from GitHub Actions")
    args = parser.parse_args()
    timestamp = args.timestamp
    print(f"Timestamp received from GitHub Actions: {timestamp}")

    # ----------------------------
    # Load dataset
    # ----------------------------
    df = pd.read_csv("gaming_mental_health.csv")
    print(df.head())

    # ----------------------------
    # Select features and target
    # ----------------------------
    features = ['age', 'screen_time', 'exercise_hours', 'face_to_face_social_hours_weekly', 'social_isolation_score', 'grades_gpa']  # adjust to your dataset
    target = 'sleep_hours'

    X = df[features]
    y = df[target]

    # ----------------------------
    # Save raw data for reproducibility
    # ----------------------------
    os.makedirs("data", exist_ok=True)
    with open("data/data.pickle", "wb") as f:
        pickle.dump(X, f)
    with open("data/target.pickle", "wb") as f:
        pickle.dump(y, f)

    # ----------------------------
    # Split dataset
    # ----------------------------
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    # ----------------------------
    # Start MLflow experiment
    # ----------------------------
    mlflow.set_tracking_uri("./mlruns")
    experiment_name = f"Sleep_Prediction_{timestamp}"
    experiment_id = mlflow.create_experiment(experiment_name)

    with mlflow.start_run(experiment_id=experiment_id, run_name=f"Sleep_Prediction_{timestamp}"):
        # Log dataset info
        mlflow.log_params({
            "dataset_name": "gaming_mental_health.csv",
            "n_samples": X.shape[0],
            "n_features": X.shape[1]
        })

        # ----------------------------
        # Train Linear Regression model
        # ----------------------------
        model = LinearRegression()
        model.fit(X_train, y_train)

        # Predict on test set
        y_pred = model.predict(X_test)

        # Log metrics
        mlflow.log_metrics({
            "R2": r2_score(y_test, y_pred),
            "MSE": mean_squared_error(y_test, y_pred)
        })
        print(f"R2 Score: {r2_score(y_test, y_pred):.4f}")
        print(f"MSE: {mean_squared_error(y_test, y_pred):.4f}")

        # ----------------------------
        # Save model
        # ----------------------------
        os.makedirs("models", exist_ok=True)
        model_filename = f"models/model_{timestamp}_lr_model.joblib"
        dump(model, model_filename)
        print(f"Model saved as {model_filename}")


        #DOES IT WORK?