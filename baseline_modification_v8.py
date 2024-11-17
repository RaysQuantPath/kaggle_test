import os
import joblib
import pandas as pd
import numpy as np
import lightgbm as lgb
import xgboost as xgb
import catboost as cbt

ROOT_DIR = './jane-street-real-time-market-data-forecasting'
TRAIN_PATH = os.path.join(ROOT_DIR, 'train.parquet')
TEST_PATH = os.path.join(ROOT_DIR, 'test.parquet')
MODEL_DIR = './models_v8'
MODEL_PATH = './pretrained_models'

os.makedirs(ROOT_DIR, exist_ok=True)
os.makedirs(MODEL_DIR, exist_ok=True)
os.makedirs(MODEL_PATH, exist_ok=True)

# Check if the training and test files exist, otherwise create placeholders
if not os.path.exists(TRAIN_PATH):
    print(f"Training file '{TRAIN_PATH}' not found. Please place the 'train.parquet' file in '{ROOT_DIR}'.")
    pd.DataFrame().to_parquet(TRAIN_PATH)

if not os.path.exists(TEST_PATH):
    print(f"Test file '{TEST_PATH}' not found. Please place the 'test.parquet' file in '{ROOT_DIR}'.")
    pd.DataFrame().to_parquet(TEST_PATH)

# Global constants
TRAINING = True
FEATURE_NAMES = [f"feature_{i:02d}" for i in range(79)]
NUM_VALID_DATES = 100
SKIP_DATES = 500
N_FOLD = 5
PURGE_WINDOW = 30
# Load training data if in training mode
if TRAINING:
    if os.path.getsize(TRAIN_PATH) > 0:
        df = pd.read_parquet(TRAIN_PATH)
        df = df[df['date_id'] >= SKIP_DATES].reset_index(drop=True)
        df = df.sort_values('date_id').reset_index(drop=True)
        dates = df['date_id'].unique()
        print("Training dataset prepared.")
    else:
        print(f"Training file '{TRAIN_PATH}' is empty. Please provide a valid training dataset.")
        exit()

# Define the weighted R² scoring function
def weighted_r2_score(y_true, y_pred, weights):
    numerator = np.sum(weights * (y_true - y_pred) ** 2)
    denominator = np.sum(weights * (y_true - np.average(y_true, weights=weights)) ** 2)
    return 1 - (numerator / denominator)

def purged_kfold_split(df, n_splits=5, purge_window=10):

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

# Model training function
def train(model_dict, model_name='lgb'):
    if TRAINING:
        X = df[FEATURE_NAMES]
        y = df['responder_6']
        w = df['weight']

        for i, (train_indices, valid_indices) in enumerate(purged_kfold_split(df, n_splits=N_FOLD, purge_window=PURGE_WINDOW)):
            X_train, X_valid = X.iloc[train_indices], X.iloc[valid_indices]
            y_train, y_valid = y.iloc[train_indices], y.iloc[valid_indices]
            w_train, w_valid = w.iloc[train_indices], w.iloc[valid_indices]

            model = model_dict[model_name]
            if model_name == 'lgb':
                model.fit(X_train, y_train, sample_weight=w_train,
                          eval_set=[(X_valid, y_valid)],
                          eval_sample_weight=[w_valid],
                          callbacks=[lgb.early_stopping(100), lgb.log_evaluation(10)])
            elif model_name == 'cbt':
                evalset = cbt.Pool(X_valid, y_valid, weight=w_valid)
                model.fit(X_train, y_train, sample_weight=w_train,
                          eval_set=[evalset],
                          early_stopping_rounds=100, verbose=10)
            elif model_name == 'xgb':
                model.fit(X_train, y_train, sample_weight=w_train,
                          eval_set=[(X_valid, y_valid)],
                          sample_weight_eval_set=[w_valid],
                          early_stopping_rounds=100, verbose=10)

            joblib.dump(model, os.path.join(MODEL_DIR, f'{model_name}_{i}.model'))
            del X_train, y_train, w_train, X_valid, y_valid, w_valid
    else:
        models.append(joblib.load(os.path.join(MODEL_PATH, f'{model_name}_{i}.model')))

# Model dictionary
model_dict = {
    'lgb': lgb.LGBMRegressor(n_estimators=500, device='gpu', gpu_use_dp=True, objective='l2'),
    'xgb': xgb.XGBRegressor(n_estimators=2000, learning_rate=0.1, max_depth=6, tree_method='hist', gpu_id=0,
                            objective='reg:squarederror'),
    'cbt': cbt.CatBoostRegressor(iterations=1000, learning_rate=0.05, task_type='GPU', loss_function='RMSE'),
}

# Train models
models = []
for model_name in model_dict.keys():
    train(model_dict, model_name)

# Evaluate the models using the test dataset
if os.path.getsize(TEST_PATH) > 0:
    test_df = pd.read_parquet(TEST_PATH)
    X_test = test_df[FEATURE_NAMES]
    y_test = test_df['responder_6']
    w_test = test_df['weight']

    model_scores = {}

    for model_name in model_dict.keys():
        fold_predictions = []
        for i in range(N_FOLD):
            model_path = os.path.join(MODEL_DIR, f'{model_name}_{i}.model')
            model = joblib.load(model_path)
            fold_predictions.append(model.predict(X_test))

        # Average predictions across folds for the current model
        y_pred = np.mean(fold_predictions, axis=0)

        # Calculate weighted R² score for the current model
        r2_score = weighted_r2_score(y_test, y_pred, w_test)
        model_scores[model_name] = r2_score
        print(f"Weighted R² score for {model_name}: {r2_score:.4f}")

    # Summary of scores
    print("\nSummary of model scores:")
    for model_name, score in model_scores.items():
        print(f"{model_name}: {score:.4f}")
else:
    print(f"Test file '{TEST_PATH}' is empty. Please provide a valid test dataset.")
