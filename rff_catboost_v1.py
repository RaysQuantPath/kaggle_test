import os
import numpy as np
import polars as pl
from catboost import CatBoostRegressor, Pool
from sklearn.kernel_approximation import RBFSampler
from sklearn.impute import SimpleImputer
import optuna
from optuna.integration import CatBoostPruningCallback

rng = np.random.default_rng(seed=42)

df_list = [
    pl.read_parquet(f'../input/jane-street-real-time-market-data-forecasting/train.parquet/partition_id={i}/part-0.parquet')
    for i in range(10)
]
df = pl.concat(df_list)

date_col = 'date_id'
feature_cols = [f'feature_{i:02d}' for i in range(0, 79)]
target_col = 'responder_6'
weight_col = 'weight'

X = df.select(feature_cols).to_pandas().values
y = df.select(target_col).to_pandas()[target_col].values
weights = df.select(weight_col).to_pandas()[weight_col].values
dates = df.select(date_col).to_pandas()[date_col].values

imputer = SimpleImputer(strategy='mean')
X_imputed = imputer.fit_transform(X)

gamma = rng.choice([0.5, 0.6, 0.7, 0.8, 0.9, 1.0])
rff = RBFSampler(gamma=gamma, n_components=500, random_state=42)
X_rff = rff.fit_transform(X_imputed) - 0.5

max_date = dates.max()
cutoff_date = max_date - 30
train_mask = dates < cutoff_date
test_mask = dates >= cutoff_date

X_train = X_rff[train_mask]
y_train = y[train_mask]
weights_train = weights[train_mask]

X_test = X_rff[test_mask]
y_test = y[test_mask]
weights_test = weights[test_mask]

def weighted_r2_score(y_true, y_pred, weights):
    numerator = np.sum(weights * (y_true - y_pred) ** 2)
    denominator = np.sum(weights * (y_true - np.average(y_true, weights=weights)) ** 2)
    return 1 - (numerator / denominator)

def objective(trial):
    params = {
        'iterations': trial.suggest_int('iterations', 100, 1000),
        'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.3),
        'depth': trial.suggest_int('depth', 4, 10),
        'l2_leaf_reg': trial.suggest_float('l2_leaf_reg', 1e-3, 1e3, log=True),
        'loss_function': 'MAE',
        'task_type': 'GPU',
        'random_state': 42,
    }

    if X_train.shape[0] != y_train.shape[0]:
        raise ValueError("Training data and labels do not match.")
    if X_test.shape[0] != y_test.shape[0]:
        raise ValueError("Test data and labels do not match.")

    train_pool = Pool(data=X_train, label=y_train, weight=weights_train)
    test_pool = Pool(data=X_test, label=y_test, weight=weights_test)

    model = CatBoostRegressor(**params, verbose=0)
    model.fit(train_pool, eval_set=test_pool, early_stopping_rounds=100,
              callbacks=[CatBoostPruningCallback(trial, 'MAE')])

    y_pred = model.predict(X_test)
    return weighted_r2_score(y_test, y_pred, weights_test)

study = optuna.create_study(direction='maximize')
study.optimize(objective, n_trials=50)

print('Best trial:')
trial = study.best_trial
print(f'  Value: {trial.value}')
print('  Params: ')
for key, value in trial.params.items():
    print(f'    {key}: {value}')

best_params = trial.params
best_params['loss_function'] = 'MAE'
best_params['task_type'] = 'GPU'
best_params['random_state'] = 42

best_model = CatBoostRegressor(**best_params, verbose=0)
best_model.fit(Pool(data=X_train, label=y_train, weight=weights_train))
y_pred = best_model.predict(X_test)
weighted_r2 = weighted_r2_score(y_test, y_pred, weights_test)
print(f'Weighted R² Score on Test Set: {weighted_r2}')
