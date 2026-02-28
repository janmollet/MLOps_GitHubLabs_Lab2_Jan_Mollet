import os
import pickle
import json
import joblib
import argparse
from sklearn.metrics import mean_squared_error, r2_score

if __name__ == '__main__':
    # ----------------------------
    # Parse timestamp
    # ----------------------------
    parser = argparse.ArgumentParser()
    parser.add_argument("--timestamp", type=str, required=True, help="Timestamp from GitHub Actions")
    args = parser.parse_args()
    timestamp = args.timestamp
    print(f"Evaluating model with timestamp: {timestamp}")

    # ----------------------------
    # Load model
    # ----------------------------
    model_filename = f'models/model_{timestamp}_lr_model.joblib'
    if not os.path.exists(model_filename):
        raise FileNotFoundError(f"Model file not found: {model_filename}")
    model = joblib.load(model_filename)

    # ----------------------------
    # Load dataset
    # ----------------------------
    data_file = 'data/data.pickle'
    target_file = 'data/target.pickle'
    if not os.path.exists(data_file) or not os.path.exists(target_file):
        raise FileNotFoundError("Data or target pickle files not found in 'data/'")

    with open(data_file, 'rb') as f:
        X = pickle.load(f)
    with open(target_file, 'rb') as f:
        y = pickle.load(f)

    # ----------------------------
    # Make predictions and compute metrics
    # ----------------------------
    y_pred = model.predict(X)
    metrics = {
        "R2": r2_score(y, y_pred),
        "MSE": mean_squared_error(y, y_pred)
    }
    print(f"Metrics: {metrics}")

    # ----------------------------
    # Save metrics to JSON
    # ----------------------------
    os.makedirs("metrics", exist_ok=True)
    metrics_file = f'metrics/{timestamp}_metrics.json'
    with open(metrics_file, 'w') as f:
        json.dump(metrics, f, indent=4)
    print(f"Metrics saved to {metrics_file}")