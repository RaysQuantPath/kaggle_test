import os
import joblib
import numpy as np
import pandas as pd
from scipy.optimize import minimize
import xgboost as xgb
import catboost as cbt
import lightgbm as lgb
import torch
from torch.utils.data import DataLoader, TensorDataset
import optuna

# 文件路径和参数
ROOT_DIR = r'C:\Users\cyg19\Desktop\kaggle_test'
TRAIN_PATH = os.path.join(ROOT_DIR, 'filtered_train.parquet')
MODEL_DIR = './models_v9_v2'
MODEL_PATH = './pretrained_models'

os.makedirs(ROOT_DIR, exist_ok=True)
os.makedirs(MODEL_DIR, exist_ok=True)
os.makedirs(MODEL_PATH, exist_ok=True)

# 全局常量
TRAINING = True
FEATURE_NAMES = [f"feature_{i:02d}" for i in range(79)] + ['symbol_id']  # 添加 'symbol_id' 到特征列表
NUM_VALID_DATES = 90
NUM_TEST_DATES = 90
SKIP_DATES = 500
N_FOLD = 5

# 加载训练数据
if TRAINING:
    if os.path.getsize(TRAIN_PATH) > 0:
        df = pd.read_parquet(TRAIN_PATH)
        df = df[df['date_id'] >= SKIP_DATES].reset_index(drop=True)
        dates = df['date_id'].unique()
        test_dates = dates[-NUM_TEST_DATES:]
        remaining_dates = dates[:-NUM_TEST_DATES]

        valid_dates = remaining_dates[-NUM_VALID_DATES:] if NUM_VALID_DATES > 0 else []
        train_dates = remaining_dates[:-NUM_VALID_DATES] if NUM_VALID_DATES > 0 else remaining_dates

        # 处理 symbol_id
        symbol_ids = df['symbol_id'].unique()
        symbol_id_to_index = {symbol_id: idx for idx, symbol_id in enumerate(symbol_ids)}
        df['symbol_id'] = df['symbol_id'].map(symbol_id_to_index)

        print("已准备好训练、验证和测试数据集。")
    else:
        print(f"训练文件 '{TRAIN_PATH}' 为空。请提供有效的训练数据集。")
        exit()

# 定义加权 R² 评分函数
def weighted_r2_score(y_true, y_pred, weights):
    numerator = np.sum(weights * (y_true - y_pred) ** 2)
    denominator = np.sum(weights * (y_true - np.average(y_true, weights=weights)) ** 2)
    return 1 - (numerator / denominator)

# ----------------- 模型调参使用 Optuna -----------------
def optimize_lgb(trial, X_train, y_train, X_valid, y_valid, w_train, w_valid):
    params = {
        'n_estimators': trial.suggest_int('n_estimators', 100, 5000),
        'learning_rate': trial.suggest_float('learning_rate', 1e-4, 0.1, log=True),
        'max_depth': trial.suggest_int('max_depth', 3, 15),
        'num_leaves': trial.suggest_int('num_leaves', 10, 100),
        'objective': 'l2',
        'device': 'gpu',
        'gpu_use_dp': True,
        'verbosity': -1
    }
    model = lgb.LGBMRegressor(**params)
    model.fit(X_train, y_train, sample_weight=w_train, eval_set=[(X_valid, y_valid)],
              sample_weight_eval_set=[w_valid],
              callbacks=[lgb.early_stopping(200)], verbose=False)
    y_pred = model.predict(X_valid)
    return -weighted_r2_score(y_valid, y_pred, w_valid)  # 负值，因为 Optuna 最小化目标

def optimize_xgb(trial, X_train, y_train, X_valid, y_valid, w_train, w_valid):
    params = {
        'n_estimators': trial.suggest_int('n_estimators', 100, 5000),
        'learning_rate': trial.suggest_float('learning_rate', 1e-4, 0.3, log=True),
        'max_depth': trial.suggest_int('max_depth', 3, 15),
        'subsample': trial.suggest_uniform('subsample', 0.5, 1.0),
        'colsample_bytree': trial.suggest_uniform('colsample_bytree', 0.5, 1.0),
        'tree_method': 'gpu_hist',
        'gpu_id': 0,
        'objective': 'reg:squarederror',
        'verbosity': 0
    }
    model = xgb.XGBRegressor(**params)
    model.fit(X_train, y_train, sample_weight=w_train, eval_set=[(X_valid, y_valid)],
              sample_weight_eval_set=[w_valid],
              early_stopping_rounds=200, verbose=False)
    y_pred = model.predict(X_valid)
    return -weighted_r2_score(y_valid, y_pred, w_valid)

def optimize_cbt(trial, X_train, y_train, X_valid, y_valid, w_train, w_valid):
    params = {
        'iterations': trial.suggest_int('iterations', 500, 5000),
        'learning_rate': trial.suggest_float('learning_rate', 1e-4, 0.3, log=True),
        'depth': trial.suggest_int('depth', 4, 10),
        'l2_leaf_reg': trial.suggest_float('l2_leaf_reg', 1e-4, 10, log=True),
        'task_type': 'GPU',
        'loss_function': 'RMSE',
        'verbose': False
    }
    model = cbt.CatBoostRegressor(**params)
    model.fit(X_train, y_train, sample_weight=w_train, eval_set=[(X_valid, y_valid)],
              sample_weight_eval_set=[w_valid],
              early_stopping_rounds=200, verbose=False)
    y_pred = model.predict(X_valid)
    return -weighted_r2_score(y_valid, y_pred, w_valid)

# 优化所有模型
studies = {}
for model_name, optimize_func in zip(['lgb', 'xgb', 'cbt'], [optimize_lgb, optimize_xgb, optimize_cbt]):
    selected_dates = [date for ii, date in enumerate(train_dates) if ii % N_FOLD != 0]
    X_train = df[FEATURE_NAMES[:-1]].loc[df['date_id'].isin(selected_dates)].values
    y_train = df['responder_6'].loc[df['date_id'].isin(selected_dates)].values
    w_train = df['weight'].loc[df['date_id'].isin(selected_dates)].values
    symbol_id_train = df['symbol_id'].loc[df['date_id'].isin(selected_dates)].values.reshape(-1, 1)
    X_train = np.hstack((X_train, symbol_id_train))

    X_valid = df[FEATURE_NAMES[:-1]].loc[df['date_id'].isin(valid_dates)].values
    y_valid = df['responder_6'].loc[df['date_id'].isin(valid_dates)].values
    w_valid = df['weight'].loc[df['date_id'].isin(valid_dates)].values
    symbol_id_valid = df['symbol_id'].loc[df['date_id'].isin(valid_dates)].values.reshape(-1, 1)
    X_valid = np.hstack((X_valid, symbol_id_valid))

    study = optuna.create_study(direction='minimize')
    study.optimize(lambda trial: optimize_func(trial, X_train, y_train, X_valid, y_valid, w_train, w_valid), n_trials=50)
    studies[model_name] = study.best_params
    print(f"{model_name} 最佳参数: ", study.best_params)

# ----------------- 模型训练函数 -----------------
def train(model_dict, model_name='lgb'):
    for i in range(N_FOLD):
        if TRAINING:
            selected_dates = [date for ii, date in enumerate(train_dates) if ii % N_FOLD != i]
            X_train = df[FEATURE_NAMES[:-1]].loc[df['date_id'].isin(selected_dates)].values
            y_train = df['responder_6'].loc[df['date_id'].isin(selected_dates)].values
            w_train = df['weight'].loc[df['date_id'].isin(selected_dates)].values
            symbol_id_train = df['symbol_id'].loc[df['date_id'].isin(selected_dates)].values.reshape(-1,1)
            X_train = np.hstack((X_train, symbol_id_train))

            if NUM_VALID_DATES > 0:
                X_valid = df[FEATURE_NAMES[:-1]].loc[df['date_id'].isin(valid_dates)].values
                y_valid = df['responder_6'].loc[df['date_id'].isin(valid_dates)].values
                w_valid = df['weight'].loc[df['date_id'].isin(valid_dates)].values
                symbol_id_valid = df['symbol_id'].loc[df['date_id'].isin(valid_dates)].values.reshape(-1,1)
                X_valid = np.hstack((X_valid, symbol_id_valid))
            else:
                X_valid, y_valid, w_valid = None, None, None

            model = model_dict[model_name]
            if model_name == 'lgb':
                model.fit(
                    X_train, y_train, sample_weight=w_train,
                    eval_set=[(X_valid, y_valid)] if NUM_VALID_DATES > 0 else None,
                    sample_weight_eval_set=[w_valid] if NUM_VALID_DATES > 0 else None,
                    callbacks=[lgb.early_stopping(200), lgb.log_evaluation(10)] if NUM_VALID_DATES > 0 else None
                )
            elif model_name == 'cbt':
                if NUM_VALID_DATES > 0:
                    evalset = cbt.Pool(X_valid, y_valid, weight=w_valid)
                    model.fit(
                        X_train, y_train, sample_weight=w_train,
                        eval_set=[evalset], early_stopping_rounds=200, verbose=10
                    )
                else:
                    model.fit(X_train, y_train, sample_weight=w_train)
            elif model_name == 'xgb':
                model.fit(
                    X_train, y_train, sample_weight=w_train,
                    eval_set=[(X_valid, y_valid)] if NUM_VALID_DATES > 0 else None,
                    sample_weight_eval_set=[w_valid] if NUM_VALID_DATES > 0 else None,
                    early_stopping_rounds=200, verbose=10
                )

            # 保存模型
            joblib.dump(model, os.path.join(MODEL_DIR, f'{model_name}_{i}.model'))
            del X_train, y_train, w_train, symbol_id_train
        else:
            models.append(joblib.load(os.path.join(MODEL_PATH, f'{model_name}_{i}.model')))

# ----------------- 获取各折预测函数 -----------------
def get_fold_predictions(model_names, df, dates, feature_names):
    fold_predictions = {model_name: [] for model_name in model_names}

    for model_name in model_names:
        for i in range(N_FOLD):
            model_path = os.path.join(MODEL_DIR, f'{model_name}_{i}.model')
            model = joblib.load(model_path)
            X = df[feature_names[:-1]].loc[df['date_id'].isin(dates)].values
            symbol_id = df['symbol_id'].loc[df['date_id'].isin(dates)].values.reshape(-1,1)
            X = np.hstack((X, symbol_id))
            fold_predictions[model_name].append(model.predict(X))

    return fold_predictions

# ----------------- 计算加权平均的预测 -----------------
def ensemble_predictions(weights, fold_predictions):
    y_pred = np.zeros_like(fold_predictions[next(iter(fold_predictions))][0])
    for idx, model_name in enumerate(fold_predictions):
        avg_pred = np.mean(fold_predictions[model_name], axis=0)
        y_pred += weights[idx] * avg_pred
    return y_pred

# ----------------- 优化权重 -----------------
def optimize_weights(fold_predictions, y_true, w_true):
    def loss_function(weights):
        weights = np.array(weights)
        weights /= weights.sum()
        y_pred = ensemble_predictions(weights, fold_predictions)
        return -weighted_r2_score(y_true, y_pred, w_true)

    initial_weights = [1 / len(fold_predictions)] * len(fold_predictions)
    bounds = [(0, 1)] * len(fold_predictions)
    result = minimize(loss_function, initial_weights, bounds=bounds, method='SLSQP')
    return result.x / result.x.sum()

# ----------------- 模型实例定义 -----------------
model_dict = {
    'lgb': lgb.LGBMRegressor(**studies['lgb']),
    'xgb': xgb.XGBRegressor(**studies['xgb']),
    'cbt': cbt.CatBoostRegressor(**studies['cbt']),
}

# ----------------- 训练和测试 -----------------
models = []
for model_name in model_dict.keys():
    train(model_dict, model_name)

# 使用测试集评估
if TRAINING:
    test_df = df[df['date_id'].isin(test_dates)]
    X_test = test_df[FEATURE_NAMES[:-1]].values
    y_test = test_df['responder_6'].values
    w_test = test_df['weight'].values
    symbol_id_test = test_df['symbol_id'].values.reshape(-1,1)
    X_test = np.hstack((X_test, symbol_id_test))

    model_names = list(model_dict.keys())
    fold_predictions = get_fold_predictions(model_names, df, test_dates, FEATURE_NAMES)

    optimized_weights = optimize_weights(fold_predictions, y_test, w_test)

    # 计算各个模型的分数
    model_scores = {}
    for model_name in model_names:
        avg_pred = np.mean(fold_predictions[model_name], axis=0)
        model_scores[model_name] = weighted_r2_score(y_test, avg_pred, w_test)
    print(f"最佳模型权重: {dict(zip(model_names, optimized_weights))}")

    y_ensemble_pred = ensemble_predictions(optimized_weights, fold_predictions)
    ensemble_r2_score = weighted_r2_score(y_test, y_ensemble_pred, w_test)
    print(f"Ensemble 的加权 R² 分数: {ensemble_r2_score:.8f}")
