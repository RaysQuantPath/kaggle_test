import os
import joblib
import pandas as pd
import numpy as np
import lightgbm as lgb
import xgboost as xgb
import catboost as cbt
from sklearn.model_selection import TimeSeriesSplit, GridSearchCV

# Define directories and file paths
ROOT_DIR = './jane-street-real-time-market-data-forecasting'
TRAIN_PATH = os.path.join(ROOT_DIR, 'train.parquet')
# TEST_PATH = os.path.join(ROOT_DIR, 'test.parquet')  # External test set is no longer needed
MODEL_DIR = './models_v6'
MODEL_PATH = './pretrained_models'

# Create necessary directories if they don't exist
os.makedirs(ROOT_DIR, exist_ok=True)
os.makedirs(MODEL_DIR, exist_ok=True)
os.makedirs(MODEL_PATH, exist_ok=True)

# Remove external test set checks
# if not os.path.exists(TEST_PATH):
#     print(f"Test file '{TEST_PATH}' not found. Please place the 'test.parquet' file in '{ROOT_DIR}'.")
#     pd.DataFrame().to_parquet(TEST_PATH)

# Global constants
TRAINING = True
FEATURE_NAMES = [f"feature_{i:02d}" for i in range(79)]
NUM_VALID_DATES = 100  # Number of dates for validation
NUM_TEST_DATES = 30  # Use the last 30 days of training data as the test set
SKIP_DATES = 500  # Number of initial dates to skip
N_FOLD = 5  # Number of outer cross-validation folds

# Load training data
if TRAINING:
    if os.path.getsize(TRAIN_PATH) > 0:
        df = pd.read_parquet(TRAIN_PATH)
        # Skip the initial SKIP_DATES
        df = df[df['date_id'] >= SKIP_DATES].reset_index(drop=True)
        dates = df['date_id'].unique()

        # Ensure there are enough dates for splitting
        if len(dates) < (NUM_VALID_DATES + NUM_TEST_DATES):
            print(
                "Not enough dates in the dataset to perform splitting. Please check the `NUM_VALID_DATES` and `NUM_TEST_DATES` settings.")
            exit()

        # Split dates into test, validation, and training sets
        test_dates = dates[-NUM_TEST_DATES:]
        remaining_dates = dates[:-NUM_TEST_DATES]

        valid_dates = remaining_dates[-NUM_VALID_DATES:] if NUM_VALID_DATES > 0 else []
        train_dates = remaining_dates[:-NUM_VALID_DATES] if NUM_VALID_DATES > 0 else remaining_dates

        print("Training, validation, and test datasets are prepared.")
    else:
        print(f"Training file '{TRAIN_PATH}' is empty. Please provide a valid training dataset.")
        exit()


# Define the weighted R² scoring function
def weighted_r2_score(y_true, y_pred, weights):
    numerator = np.sum(weights * (y_true - y_pred) ** 2)
    denominator = np.sum(weights * (y_true - np.average(y_true, weights=weights)) ** 2)
    return 1 - (numerator / denominator)


# Model training function with nested cross-validation and hyperparameter tuning
def train(model_dict, model_name='lgb'):
    if TRAINING:
        # Extract training data based on train_dates
        train_df = df[df['date_id'].isin(train_dates)]
        X = train_df[FEATURE_NAMES].reset_index(drop=True)
        y = train_df['responder_6'].reset_index(drop=True)
        w = train_df['weight'].reset_index(drop=True)

        # Initialize outer cross-validation
        outer_cv = TimeSeriesSplit(n_splits=N_FOLD)

        for i, (train_index, valid_index) in enumerate(outer_cv.split(X)):
            print(f"Starting outer fold {i + 1}/{N_FOLD}")
            X_train_outer, X_valid_outer = X.iloc[train_index], X.iloc[valid_index]
            y_train_outer, y_valid_outer = y.iloc[train_index], y.iloc[valid_index]
            w_train_outer, w_valid_outer = w.iloc[train_index], w.iloc[valid_index]

            # Define hyperparameter grid for GridSearchCV
            param_grid = {
                'n_estimators': [100, 500],
                'learning_rate': [0.01, 0.1],
                'max_depth': [3, 6]
            }

            # Initialize inner cross-validation
            inner_cv = TimeSeriesSplit(n_splits=3)

            # Initialize the model
            model = model_dict[model_name]

            # Initialize GridSearchCV with TimeSeriesSplit
            grid_search = GridSearchCV(
                estimator=model,
                param_grid=param_grid,
                scoring='neg_mean_squared_error',
                cv=inner_cv,
                n_jobs=-1
            )

            # Fit GridSearchCV
            grid_search.fit(
                X_train_outer, y_train_outer,
                sample_weight=w_train_outer,
                eval_set=[(X_valid_outer, y_valid_outer)],
                eval_sample_weight=[w_valid_outer],
                callbacks=[lgb.early_stopping(100), lgb.log_evaluation(10)] if model_name == 'lgb' else []
            )

            # Retrieve the best model from GridSearchCV
            best_model = grid_search.best_estimator_
            print(f"Best parameters for fold {i + 1}: {grid_search.best_params_}")

            # Predict on the validation set
            y_pred = best_model.predict(X_valid_outer)
            r2 = weighted_r2_score(y_valid_outer, y_pred, w_valid_outer)
            print(f"Fold {i + 1}, Weighted R² Score: {r2:.4f}")

            # Save the best model
            model_filename = f'{model_name}_{i}.model'
            joblib.dump(best_model, os.path.join(MODEL_DIR, model_filename))
            print(f"Model saved to {os.path.join(MODEL_DIR, model_filename)}\n")

            # Clean up memory
            del X_train_outer, y_train_outer, w_train_outer, X_valid_outer, y_valid_outer, w_valid_outer
    else:
        # Load pre-trained models if not training
        for i in range(N_FOLD):
            model_filename = f'{model_name}_{i}.model'
            model_path = os.path.join(MODEL_PATH, model_filename)
            if os.path.exists(model_path):
                models.append(joblib.load(model_path))
                print(f"Loaded model from {model_path}")
            else:
                print(f"Model file '{model_path}' does not exist. Skipping this fold.")


# Model dictionary with configurations
model_dict = {
    'lgb': lgb.LGBMRegressor(
        n_estimators=500,
        device='gpu',
        gpu_use_dp=True,
        objective='l2'
    ),
    'xgb': xgb.XGBRegressor(
        n_estimators=2000,
        learning_rate=0.1,
        max_depth=6,
        tree_method='hist',
        gpu_id=0,
        objective='reg:squarederror'
    ),
    'cbt': cbt.CatBoostRegressor(
        iterations=1000,
        learning_rate=0.05,
        task_type='GPU',
        loss_function='RMSE'
    ),
}

# Train models
models = []
for model_name in model_dict.keys():
    print(f"Training model: {model_name}")
    train(model_dict, model_name)
    print(f"Finished training model: {model_name}\n")

# Evaluate the models using the internal test dataset
if TRAINING:
    # Use the last NUM_TEST_DATES as the test set
    test_df = df[df['date_id'].isin(test_dates)]
    if test_df.empty:
        print("Test set is empty. Please check the `NUM_TEST_DATES` setting.")
        exit()

    X_test = test_df[FEATURE_NAMES]
    y_test = test_df['responder_6']
    w_test = test_df['weight']

    model_scores = {}

    for model_name in model_dict.keys():
        fold_predictions = []
        for i in range(N_FOLD):
            model_filename = f'{model_name}_{i}.model'
            model_path = os.path.join(MODEL_DIR, model_filename)
            if not os.path.exists(model_path):
                print(f"Model file '{model_path}' does not exist. Skipping this fold.")
                continue
            model = joblib.load(model_path)
            preds = model.predict(X_test)
            fold_predictions.append(preds)
            print(f"Loaded predictions from fold {i + 1} for model {model_name}")

        if not fold_predictions:
            print(f"No valid predictions available to calculate the score for {model_name}.")
            continue

        # Average predictions across folds for the current model
        y_pred = np.mean(fold_predictions, axis=0)

        # Calculate weighted R² score
        r2_score = weighted_r2_score(y_test, y_pred, w_test)
        model_scores[model_name] = r2_score
        print(f"Weighted R² score for {model_name}: {r2_score:.8f}")

    # Summary of scores
    print("\nSummary of model scores:")
    for model_name, score in model_scores.items():
        print(f"{model_name}: {score:.8f}")
else:
    print("No test dataset available for evaluation.")
