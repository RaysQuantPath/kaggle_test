import os
import numpy as np
import polars as pl
from catboost import CatBoostRegressor, Pool
from sklearn.kernel_approximation import RBFSampler
from sklearn.impute import SimpleImputer

rng = np.random.default_rng(seed=42)

df = pl.read_parquet('filtered_train.parquet')

date_col = 'date_id'
symbol_col = 'symbol_id'
feature_cols = [f'feature_{i:02d}' for i in range(0, 79)]
target_col = 'responder_6'
weight_col = 'weight'

X = df.select(feature_cols).to_pandas().values
y = df.select(target_col).to_pandas()[target_col].values
weights = df.select(weight_col).to_pandas()[weight_col].values
dates = df.select(date_col).to_pandas()[date_col].values
symbols = df.select(symbol_col).to_pandas()[symbol_col].values  # 新增

imputer = SimpleImputer(strategy='mean')
X_imputed = imputer.fit_transform(X)

gamma = rng.choice([0.5, 0.6, 0.7, 0.8, 0.9, 1.0])
rff = RBFSampler(gamma=gamma, n_components=500, random_state=42)
X_rff = rff.fit_transform(X_imputed) - 0.5

unique_symbols = np.unique(symbols)

y_test_all = []
y_pred_all = []
weights_test_all = []

def weighted_r2_score(y_true, y_pred, weights):
    numerator = np.sum(weights * (y_true - y_pred) ** 2)
    denominator = np.sum(weights * (y_true - np.average(y_true, weights=weights)) ** 2)
    return 1 - (numerator / denominator)

class R2MetricCBT:
    def get_final_error(self, error, weight):
        return 1 - error / (weight + 1e-38)

    def is_max_optimal(self):
        return True

    def evaluate(self, approxes, target, weight):
        assert len(approxes) == 1
        assert len(target) == len(approxes[0])

        approx = approxes[0]

        error_sum = 0.0
        weight_sum = 0.0

        for i in range(len(approx)):
            w = 1.0 if weight is None else weight[i]
            weight_sum += w * (target[i] ** 2)
            error_sum += w * ((approx[i] - target[i]) ** 2)

        return error_sum, weight_sum

    def get_metric_name(self):
        return 'r2'

params = {
    'iterations': 1000,
    'learning_rate': 0.1,
    'depth': 6,
    'l2_leaf_reg': 3,
    'loss_function': 'RMSE',
    'task_type': 'GPU',
    'random_state': 42,
    'eval_metric': R2MetricCBT(),
    'early_stopping_rounds': 100,
    'verbose': False,
}

for symbol in unique_symbols:
    symbol_mask = symbols == symbol
    X_symbol = X_rff[symbol_mask]
    y_symbol = y[symbol_mask]
    weights_symbol = weights[symbol_mask]
    dates_symbol = dates[symbol_mask]

    max_date = dates_symbol.max()
    cutoff_date = max_date - 30
    train_mask = dates_symbol < cutoff_date
    test_mask = dates_symbol >= cutoff_date

    if not np.any(test_mask):
        continue

    X_train = X_symbol[train_mask]
    y_train = y_symbol[train_mask]
    weights_train = weights_symbol[train_mask]

    X_test = X_symbol[test_mask]
    y_test = y_symbol[test_mask]
    weights_test = weights_symbol[test_mask]

    if X_train.shape[0] == 0 or X_test.shape[0] == 0:
        continue

    train_pool = Pool(data=X_train, label=y_train, weight=weights_train)
    validation_pool = Pool(data=X_test, label=y_test, weight=weights_test)

    model = CatBoostRegressor(**params)
    model.fit(
        train_pool,
        eval_set=validation_pool
    )

    y_pred = model.predict(X_test)
    y_test_all.append(y_test)
    y_pred_all.append(y_pred)
    weights_test_all.append(weights_test)

y_test_all = np.concatenate(y_test_all)
y_pred_all = np.concatenate(y_pred_all)
weights_test_all = np.concatenate(weights_test_all)

weighted_r2 = weighted_r2_score(y_test_all, y_pred_all, weights_test_all)
print(f'Weighted R² Score on Test Set: {weighted_r2}')
