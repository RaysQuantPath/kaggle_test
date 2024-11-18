import os
import joblib
import pandas as pd
import numpy as np
import lightgbm as lgb
import xgboost as xgb
import catboost as cbt
from sklearn.model_selection import TimeSeriesSplit

ROOT_DIR = './jane-street-real-time-market-data-forecasting'
TRAIN_PATH = os.path.join(ROOT_DIR, 'train.parquet')
# TEST_PATH = os.path.join(ROOT_DIR, 'test.parquet')  # 不再需要外部测试集
MODEL_DIR = './models_v2'
MODEL_PATH = './pretrained_models'

os.makedirs(ROOT_DIR, exist_ok=True)
os.makedirs(MODEL_DIR, exist_ok=True)
os.makedirs(MODEL_PATH, exist_ok=True)

# 移除外部测试集的检查
# if not os.path.exists(TEST_PATH):
#     print(f"Test file '{TEST_PATH}' not found. Please place the 'test.parquet' file in '{ROOT_DIR}'.")
#     pd.DataFrame().to_parquet(TEST_PATH)

# 全局常量
TRAINING = True
FEATURE_NAMES = [f"feature_{i:02d}" for i in range(79)]
NUM_VALID_DATES = 100  # 可根据需要调整
NUM_TEST_DATES = 30  # 使用训练集最后30天作为测试集
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

        print("已准备好训练、验证和测试数据集。")
    else:
        print(f"训练文件 '{TRAIN_PATH}' 为空。请提供有效的训练数据集。")
        exit()

# 定义加权R²评分函数
def weighted_r2_score(y_true, y_pred, weights):
    numerator = np.sum(weights * (y_true - y_pred) ** 2)
    denominator = np.sum(weights * (y_true - np.average(y_true, weights=weights)) ** 2)
    return 1 - (numerator / denominator)

# 模型训练函数
def train(model_dict, model_name='lgb'):
    if TRAINING:
        # 准备训练数据
        train_df = df[df['date_id'].isin(train_dates)]
        X_train_full = train_df[FEATURE_NAMES]
        y_train_full = train_df['responder_6']
        w_train_full = train_df['weight']

        tscv = TimeSeriesSplit(n_splits=N_FOLD)
        for i, (train_index, valid_index) in enumerate(tscv.split(X_train_full)):
            X_train = X_train_full.iloc[train_index]
            X_valid = X_train_full.iloc[valid_index]
            y_train = y_train_full.iloc[train_index]
            y_valid = y_train_full.iloc[valid_index]
            w_train = w_train_full.iloc[train_index]
            w_valid = w_train_full.iloc[valid_index]

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

# 模型字典
model_dict = {
    'lgb': lgb.LGBMRegressor(n_estimators=500, device='gpu', gpu_use_dp=True, objective='l2'),
    'xgb': xgb.XGBRegressor(n_estimators=2000, learning_rate=0.1, max_depth=6, tree_method='hist', gpu_id=0,
                            objective='reg:squarederror'),
    'cbt': cbt.CatBoostRegressor(iterations=1000, learning_rate=0.05, task_type='GPU', loss_function='RMSE'),
}

# 训练模型
models = []
for model_name in model_dict.keys():
    train(model_dict, model_name)

# 使用测试集评估模型
if TRAINING:
    test_df = df[df['date_id'].isin(test_dates)]
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

        # 对当前模型的各折预测取平均值
        y_pred = np.mean(fold_predictions, axis=0)

        # 计算加权 R² 分数
        r2_score = weighted_r2_score(y_test, y_pred, w_test)
        model_scores[model_name] = r2_score
        print(f"{model_name} 的加权 R² 分数: {r2_score:.8f}")

    # 分数摘要
    print("\n模型分数摘要:")
    for model_name, score in model_scores.items():
        print(f"{model_name}: {score:.8f}")
else:
    print("没有可用于评估的测试数据集。")
