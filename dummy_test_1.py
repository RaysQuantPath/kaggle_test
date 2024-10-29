# PyTorch Imports
import numpy as np
import polars as pl  # Ensure polars is imported
from catboost import CatBoostRegressor, Pool
from datetime import timedelta

# Read and concatenate parquet files
df_list = [
    pl.read_parquet(f'../input/jane-street-real-time-market-data-forecasting/train.parquet/partition_id={i}/part-0.parquet')
    for i in range(10)
]

# Combine DataFrames
df = pl.concat(df_list)

# Assume the date column is named 'date_id'; modify if necessary
date_col = 'date_id'

# Convert date column to datetime type
df = df.with_columns([
    pl.col(date_col).cast(pl.Datetime).alias(date_col)
])

# Sort by date
df = df.sort(date_col)

# Determine the cutoff date (30 days before the maximum date)
max_date = df.select(pl.col(date_col).max()).to_numpy()[0][0]
cutoff_date = max_date - 30

# Split into training and testing sets
train_df = df.filter(pl.col(date_col) < cutoff_date)
test_df = df.filter(pl.col(date_col) >= cutoff_date)

# Define feature, target, and weight columns
feature_cols = [f'feature_{i:02d}' for i in range(0, 79)]
target_col = 'responder_6'
weight_col = 'weight'

# Prepare training data
X_train = train_df.select(feature_cols).to_numpy()
y_train = train_df.select(target_col).to_numpy().ravel()
weights_train = train_df.select(weight_col).to_numpy().ravel()

# Create a Pool for training
train_pool = Pool(data=X_train, label=y_train, weight=weights_train)

# Define the CatBoost model
model = CatBoostRegressor(
    iterations=500,
    learning_rate=0.05,
    depth=6,
    loss_function='MAE',
    task_type="GPU",  # Enable GPU if available
    use_best_model=False,
    verbose=100
)

# Train the model all at once
model.fit(train_pool)

# Define weighted R² score function
def weighted_r2_score(y_true, y_pred, weights):
    numerator = np.sum(weights * (y_true - y_pred) ** 2)
    denominator = np.sum(weights * (y_true - np.average(y_true, weights=weights)) ** 2)
    r2 = 1 - (numerator / denominator)
    return r2

# Prepare test data
X_test = test_df.select(feature_cols).to_numpy()
y_test = test_df.select(target_col).to_numpy().ravel()
weights_test = test_df.select(weight_col).to_numpy().ravel()

# Predict on test data
y_pred = model.predict(X_test)

# Calculate weighted R² score
weighted_r2 = weighted_r2_score(y_test, y_pred, weights_test)
print(f'Weighted R² Score on Test Set: {weighted_r2}')
