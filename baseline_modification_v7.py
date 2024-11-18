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
MODEL_DIR = './models_v7'
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

        # Calculate volatility for each date_id
        volatility = df.groupby('date_id')['responder_6'].std().reset_index()
        volatility.columns = ['date_id', 'volatility']

        # Merge volatility back to the main dataframe
        df = df.merge(volatility, on='date_id', how='left')

        # Assign each date_id to a volatility group using quantiles
        df['vol_group'] = pd.qcut(df['volatility'], q=N_FOLD, labels=False)

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
        groups = train_df['vol_group'].reset_index(drop=True)  # Use volatility groups for cross-validation

        # Initialize GroupKFold to ensure each fold has distinct volatility groups
        from sklearn.model_selection import GroupKFold
        gkf = GroupKFold(n_splits=N_FOLD)

        for i, (train_idx, valid_idx) in enumerate(gkf.split(X, y, groups)):
            print(f"Training fold {i + 1}/{N_FOLD} for model: {model_name}")
            X_train, X_valid = X.iloc[train_idx], X.iloc[valid_idx]
            y_train, y_valid = y.iloc[train_idx], y.iloc[valid_idx]
            w_train, w_valid = w.iloc[train_idx], w.iloc[valid_idx]

            # Define hyperparameter grid for GridSearchCV
            param_grid = {
                'n_estimators': [100, 500],
                'learning_rate': [0.01, 0.1],
                'max_depth': [3, 6] if model_name != 'cbt' else [6, 10],  # CatBoost may benefit from different depths
            }

            # Initialize inner cross-validation for hyperparameter tuning
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
                early_stopping_rounds=100,
                verbose=0
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
for model_name in model_dict.keys():
    train(model_dict, model_name)

# Evaluate the models using the internal test dataset
if TRAINING:
    # Use the last 30 days of training data as the test set
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
