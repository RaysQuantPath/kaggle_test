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
# TEST_PATH is no longer needed as we will use an internal test set
MODEL_DIR = './models_v8'
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
N_FOLD = 5  # Number of cross-validation folds
PURGE_WINDOW = 30  # Number of days to purge around the validation set to prevent leakage

# Load training data
if TRAINING:
    if os.path.getsize(TRAIN_PATH) > 0:
        df = pd.read_parquet(TRAIN_PATH)
        # Skip the initial SKIP_DATES
        df = df[df['date_id'] >= SKIP_DATES].reset_index(drop=True)
        # Sort the dataframe by date_id to ensure temporal order
        df = df.sort_values('date_id').reset_index(drop=True)
        dates = df['date_id'].unique()

        # Ensure there are enough dates for splitting
        if len(dates) < (NUM_VALID_DATES + NUM_TEST_DATES):
            print(
                "Not enough dates in the dataset to perform splitting. Please check the `NUM_VALID_DATES` and `NUM_TEST_DATES` settings.")
            exit()

        # Split dates into test and remaining sets
        test_dates = dates[-NUM_TEST_DATES:]
        remaining_dates = dates[:-NUM_TEST_DATES]

        # Further split remaining dates into training and validation sets
        valid_dates = remaining_dates[-NUM_VALID_DATES:] if NUM_VALID_DATES > 0 else []
        train_dates = remaining_dates[:-NUM_VALID_DATES] if NUM_VALID_DATES > 0 else remaining_dates

        print("Training and validation datasets prepared.")
    else:
        print(f"Training file '{TRAIN_PATH}' is empty. Please provide a valid training dataset.")
        exit()


# Define the weighted R² scoring function
def weighted_r2_score(y_true, y_pred, weights):
    numerator = np.sum(weights * (y_true - y_pred) ** 2)
    denominator = np.sum(weights * (y_true - np.average(y_true, weights=weights)) ** 2)
    return 1 - (numerator / denominator)


# Custom K-Fold splitter with purge window to prevent data leakage
def purged_kfold_split(df, n_splits=5, purge_window=30):
    unique_dates = df['date_id'].unique()
    n_dates = len(unique_dates)
    fold_size = n_dates // n_splits

    date_folds = []
    for i in range(n_splits):
        start = i * fold_size
        if i == n_splits - 1:
            end = n_dates
        else:
            end = (i + 1) * fold_size
        test_dates = unique_dates[start:end]
        date_folds.append(test_dates)

    for i in range(n_splits):
        test_dates = date_folds[i]
        test_start_date = test_dates[0]
        test_end_date = test_dates[-1]

        train_dates = np.concatenate([date_folds[j] for j in range(n_splits) if j != i])

        purge_start_date = test_start_date - purge_window
        purge_end_date = test_end_date + purge_window

        purge_start_date = max(purge_start_date, unique_dates[0])
        purge_end_date = min(purge_end_date, unique_dates[-1])

        purge_dates = unique_dates[(unique_dates >= purge_start_date) & (unique_dates <= purge_end_date)]
        train_dates = np.setdiff1d(train_dates, purge_dates)

        train_indices = df[df['date_id'].isin(train_dates)].index.values
        test_indices = df[df['date_id'].isin(test_dates)].index.values

        yield train_indices, test_indices


# Model training function with nested cross-validation and hyperparameter tuning
def train(model_dict, model_name='lgb'):
    if TRAINING:
        X = df[FEATURE_NAMES]
        y = df['responder_6']
        w = df['weight']

        for i, (train_indices, valid_indices) in enumerate(
                purged_kfold_split(df, n_splits=N_FOLD, purge_window=PURGE_WINDOW)):
            print(f"Training fold {i + 1}/{N_FOLD} for model: {model_name}")
            X_train, X_valid = X.iloc[train_indices], X.iloc[valid_indices]
            y_train, y_valid = y.iloc[train_indices], y.iloc[valid_indices]
            w_train, w_valid = w.iloc[train_indices], w.iloc[valid_indices]

            # Define hyperparameter grid for GridSearchCV
            param_grid = {
                'n_estimators': [100, 500],
                'learning_rate': [0.01, 0.1],
                'max_depth': [3, 6]
            }

            # Initialize TimeSeriesSplit for inner cross-validation
            inner_cv = TimeSeriesSplit(n_splits=3)

            # Initialize GridSearchCV
            grid_search = GridSearchCV(
                estimator=model_dict[model_name],
                param_grid=param_grid,
                scoring='neg_mean_squared_error',
                cv=inner_cv,
                n_jobs=-1,
                verbose=0
            )

            # Fit GridSearchCV
            grid_search.fit(
                X_train, y_train,
                sample_weight=w_train,
                eval_set=[(X_valid, y_valid)],
                eval_sample_weight=[w_valid],
                callbacks=[lgb.early_stopping(100), lgb.log_evaluation(10)] if model_name == 'lgb' else []
            )

            # Retrieve the best estimator
            best_model = grid_search.best_estimator_
            print(f"Best parameters for fold {i + 1}: {grid_search.best_params_}")

            # Predict on the validation set
            y_pred = best_model.predict(X_valid)
            r2 = weighted_r2_score(y_valid, y_pred, w_valid)
            print(f"Fold {i + 1} Weighted R² Score: {r2:.4f}")

            # Save the best model for this fold
            model_filename = os.path.join(MODEL_DIR, f'{model_name}_{i}.model')
            joblib.dump(best_model, model_filename)
            print(f"Model saved to {model_filename}\n")

            # Clean up memory
            del X_train, y_train, w_train, X_valid, y_valid, w_valid, best_model
    else:
        # Load pre-trained models if not training
        for i in range(N_FOLD):
            model_filename = os.path.join(MODEL_PATH, f'{model_name}_{i}.model')
            if os.path.exists(model_filename):
                models.append(joblib.load(model_filename))
                print(f"Loaded model from {model_filename}")
            else:
                print(f"Model file '{model_filename}' does not exist. Skipping this fold.")


# Model dictionary with configurations
model_dict = {
    'lgb': lgb.LGBMRegressor(
        n_estimators=500,
        device='gpu',
        gpu_use_dp=True,
        objective='l2',
        verbose=-1
    ),
    'xgb': xgb.XGBRegressor(
        n_estimators=2000,
        learning_rate=0.1,
        max_depth=6,
        tree_method='hist',
        gpu_id=0,
        objective='reg:squarederror',
        verbosity=0
    ),
    'cbt': cbt.CatBoostRegressor(
        iterations=1000,
        learning_rate=0.05,
        task_type='GPU',
        loss_function='RMSE',
        verbose=0
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
            model_path = os.path.join(MODEL_DIR, f'{model_name}_{i}.model')
            if not os.path.exists(model_path):
                print(f"Model file '{model_path}' does not exist. Skipping this fold.")
                continue
            model = joblib.load(model_path)
            y_pred_fold = model.predict(X_test)
            fold_predictions.append(y_pred_fold)
            del model  # Free up memory

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
