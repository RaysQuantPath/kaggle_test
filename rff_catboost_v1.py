import os
import numpy as np
import polars as pl
from catboost import CatBoostRegressor, Pool
from sklearn.kernel_approximation import RBFSampler
from datetime import timedelta
import optuna
from optuna.integration import CatBoostPruningCallback

rng = np.random.default_rng(seed=42)

df_list = [
    pl.read_parquet(f'../input/jane-street-real-time-market-data-forecasting/train.parquet/partition_id={i}/part-0.parquet')
    for i in range(10)
]
df = pl.concat(df_list)

date_col = 'date_id'
df = df.with_columns([pl.col(date_col).cast(pl.Datetime).alias(date_col)])
df = df.sort(date_col)

feature_cols = [f'feature_{i:02d}' for i in range(0, 79)]
target_col = 'responder_6'
weight_col = 'weight'

n_rff_features = 500
gamma = rng.choice([0.5, 0.6, 0.7, 0.8, 0.9, 1.0])

rff = RBFSampler(gamma=gamma, n_components=n_rff_features, random_state=42)

max_date = df.select(pl.col(date_col).max()).to_numpy()[0][0]
cutoff_date = max_date - 30
train_df = df.filter(pl.col(date_col) < cutoff_date)
test_df = df.filter(pl.col(date_col) >= cutoff_date)

X_train_original = train_df.select(feature_cols).to_numpy()
X_train_rff = rff.fit_transform(X_train_original)
X_train_rff = X_train_rff - 0.5
y_train = train_df.select(target_col).to_numpy().ravel()
weights_train = train_df.select(weight_col).to_numpy().ravel()

X_test_original = test_df.select(feature_cols).to_numpy()
X_test_rff = rff.transform(X_test_original)
X_test_rff = X_test_rff - 0.5
y_test = test_df.select(target_col).to_numpy().ravel()
weights_test = test_df.select(weight_col).to_numpy().ravel()

def weighted_r2_score(y_true, y_pred, weights):
    numerator = np.sum(weights * (y_true - y_pred) ** 2)
    denominator = np.sum(weights * (y_true - np.average(y_true, weights=weights)) ** 2)
    r2 = 1 - (numerator / denominator)
    return r2

def objective(trial):
    params = {
        'iterations': trial.suggest_int('iterations', 100, 1000),
        'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.3),
        'depth': trial.suggest_int('depth', 4, 10),
        'l2_leaf_reg': trial.suggest_float('l2_leaf_reg', 1e-3, 1e3, log=True),
        'loss_function': 'MAE',
        'task_type': 'GPU',
    }

    train_pool = Pool(data=X_train_rff, label=y_train, weight=weights_train)
    test_pool = Pool(data=X_test_rff, label=y_test, weight=weights_test)

    model = CatBoostRegressor(**params, verbose=0)
    model.fit(train_pool, eval_set=test_pool, verbose=0,
              early_stopping_rounds=100, callbacks=[CatBoostPruningCallback(trial, 'MAE')])

    y_pred = model.predict(X_test_rff)
    weighted_r2 = weighted_r2_score(y_test, y_pred, weights_test)

    return weighted_r2

study = optuna.create_study(direction='maximize')
study.optimize(objective, n_trials=50)

print('Best trial:')
trial = study.best_trial
print(f'  Value: {trial.value}')
print('  Params: ')
for key, value in trial.params.items():
    print(f'    {key}: {value}')

y_pred = CatBoostRegressor(**trial.params, verbose=0).fit(Pool(data=X_train_rff, label=y_train, weight=weights_train)).predict(X_test_rff)
weighted_r2 = weighted_r2_score(y_test, y_pred, weights_test)
print(f'Weighted R² Score on Test Set: {weighted_r2}')
