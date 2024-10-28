# PyTorch Imports
import numpy as np
import polars as pl  # 确保已导入polars
from catboost import CatBoostRegressor, Pool
from datetime import timedelta

df_list = [
    pl.read_parquet(f'../input/jane-street-real-time-market-data-forecasting/train.parquet/partition_id={i}/part-0.parquet')
    for i in range(10)
]

# 合并 DataFrame
df = pl.concat(df_list)

# 假设日期列名为 'timestamp'，请根据实际情况修改
date_col = 'date_id'

df = df.with_columns([
    pl.col(date_col).cast(pl.Datetime).alias(date_col)
])
# 按日期排序
df = df.sort(date_col)

max_date = df.select(pl.col(date_col).max()).to_numpy()[0][0]

cutoff_date = max_date - timedelta(days=30)

train_df = df.filter(pl.col(date_col) < cutoff_date)
test_df = df.filter(pl.col(date_col) >= cutoff_date)

feature_cols = [f'feature_{i:02d}' for i in range(0, 79)]
target_col = 'responder_6'
weight_col = 'weight'

# 定义CatBoost模型
model = CatBoostRegressor(
    iterations=500,
    learning_rate=0.05,
    depth=6,
    loss_function='RMSE',
    task_type="GPU",  # 启用GPU
    use_best_model=False,
    verbose=100
)

# 分批训练模型
batch_size = 100000
num_rows = train_df.shape[0]  # 训练集的总行数

for start in range(0, num_rows, batch_size):
    end = min(start + batch_size, num_rows)

    batch = train_df[start:end]
    X_batch = batch.select(feature_cols).to_numpy()
    y_batch = batch.select(target_col).to_numpy().ravel()
    weights_batch = batch.select(weight_col).to_numpy().ravel()

    train_pool = Pool(data=X_batch, label=y_batch, weight=weights_batch)

    model.fit(train_pool, init_model=model if start > 0 else None)

def weighted_r2_score(y_true, y_pred, weights):
    numerator = np.sum(weights * (y_true - y_pred) ** 2)
    denominator = np.sum(weights * (y_true - np.average(y_true, weights=weights)) ** 2)
    r2 = 1 - (numerator / denominator)
    return r2

X_test = test_df.select(feature_cols).to_numpy()
y_test = test_df.select(target_col).to_numpy().ravel()
weights_test = test_df.select(weight_col).to_numpy().ravel()

test_pool = Pool(data=X_test, label=y_test, weight=weights_test)

y_pred = model.predict(X_test)

weighted_r2 = weighted_r2_score(y_test, y_pred, weights_test)
print(f'Weighted R² Score on Test Set: {weighted_r2}')
