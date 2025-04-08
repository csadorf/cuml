import xgboost as xgb
import numpy as np
import tempfile
import os

def build_fil_classifier():
    # Create minimal training data
    train_data = np.random.rand(100, 10)  # 100 samples, 10 features
    train_label = np.random.randint(0, 2, 100)  # Binary labels

    # Create DMatrix
    dtrain = xgb.DMatrix(train_data, label=train_label)

    # Basic parameters
    params = {
        "eval_metric": "error",
        "objective": "binary:logistic",
        "device": "cuda",
        "max_depth": 3,
    }

    # This should trigger the error
    bst = xgb.train(params, dtrain, num_boost_round=10)
    return bst

if __name__ == "__main__":
    try:
        model = build_fil_classifier()
        print("Model trained successfully")
    except Exception as e:
        print(f"Error occurred: {str(e)}")
