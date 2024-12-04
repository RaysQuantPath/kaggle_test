import os
import joblib
import numpy as np
import pandas as pd
from scipy.optimize import minimize
import xgboost as xgb
import catboost as cbt
import lightgbm as lgb
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
import warnings
import copy

# 忽略某些警告
warnings.filterwarnings("ignore")

# 文件路径和参数
ROOT_DIR = r'C:\Users\cyg19\Desktop\kaggle_test'
TRAIN_PATH = os.path.join(ROOT_DIR, 'filtered_train.parquet')
MODEL_DIR = './models_v15_v2'
MODEL_PATH = './pretrained_models'

os.makedirs(ROOT_DIR, exist_ok=True)
os.makedirs(MODEL_DIR, exist_ok=True)
os.makedirs(MODEL_PATH, exist_ok=True)

# 全局常量
TRAINING = True
FEATURE_NAMES = [f"feature_{i:02d}" for i in range(79)] + ['symbol_id']
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
        num_symbols = len(symbol_ids)
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

# ----------------- HFLGB 模型 -----------------
class HFLGBModel:
    def __init__(self, **params):
        self.params = params
        self.model = None
        # 数据预处理器
        self.imputer = SimpleImputer(strategy='mean')
        self.scaler = StandardScaler()
        self.all_missing_cols = []

    def fit(self, X_train, y_train, X_valid=None, y_valid=None, sample_weight=None, eval_sample_weight=None):
        # 转换为 DataFrame 以便处理
        X_train_df = pd.DataFrame(X_train, columns=FEATURE_NAMES)
        X_valid_df = pd.DataFrame(X_valid, columns=FEATURE_NAMES) if X_valid is not None else None

        # 识别所有值缺失的列
        self.all_missing_cols = X_train_df.columns[X_train_df.isna().all()].tolist()

        # 填充这些列为0
        if self.all_missing_cols:
            X_train_df[self.all_missing_cols] = 0
            if X_valid_df is not None:
                X_valid_df[self.all_missing_cols] = 0

        # 对所有特征进行均值填充
        X_train_imputed = self.imputer.fit_transform(X_train_df)
        if X_valid_df is not None:
            X_valid_imputed = self.imputer.transform(X_valid_df)
        else:
            X_valid_imputed = None

        # 标准化数据
        X_train_scaled = self.scaler.fit_transform(X_train_imputed)
        if X_valid_imputed is not None:
            X_valid_scaled = self.scaler.transform(X_valid_imputed)
        else:
            X_valid_scaled = None

        # 转换为 DataFrame，便于处理 date_id
        y_train_df = pd.DataFrame({'label': y_train}, index=X_train_df.index)
        if X_valid_scaled is not None:
            y_valid_df = pd.DataFrame({'label': y_valid}, index=X_valid_df.index)
        else:
            y_valid_df = None

        # 从 df 中获取 date_id，用于计算去均值
        date_id_train = df.loc[X_train_df.index, 'date_id']
        if X_valid_scaled is not None:
            date_id_valid = df.loc[X_valid_df.index, 'date_id']
        else:
            date_id_valid = None

        # 按照 date_id 进行去均值
        y_train_df['date_id'] = date_id_train.values
        y_train_df['label'] = y_train_df['label'] - y_train_df.groupby('date_id')['label'].transform('mean')

        if y_valid_df is not None:
            y_valid_df['date_id'] = date_id_valid.values
            y_valid_df['label'] = y_valid_df['label'] - y_valid_df.groupby('date_id')['label'].transform('mean')

        # 将 label 转换为二分类标签
        y_train_binary = (y_train_df['label'] > 0).astype(int).values
        if y_valid_df is not None:
            y_valid_binary = (y_valid_df['label'] > 0).astype(int).values
        else:
            y_valid_binary = None

        # 创建 LightGBM 数据集
        lgb_train = lgb.Dataset(X_train_scaled, label=y_train_binary, weight=sample_weight)
        if X_valid_scaled is not None and y_valid_binary is not None:
            lgb_valid = lgb.Dataset(X_valid_scaled, label=y_valid_binary, weight=eval_sample_weight, reference=lgb_train)
        else:
            lgb_valid = None

        # 设置参数
        params = self.params.copy()
        params.setdefault('objective', 'binary')
        params.setdefault('device', 'gpu')
        params.setdefault('gpu_use_dp', True)
        params.setdefault('verbosity', -1)

        # 设置回调函数
        evals_result = {}
        callbacks = [lgb.log_evaluation(period=10), lgb.record_evaluation(evals_result)]
        if lgb_valid is not None:
            callbacks.append(lgb.early_stopping(stopping_rounds=200))

        # 训练模型
        if lgb_valid is not None:
            self.model = lgb.train(
                params,
                lgb_train,
                valid_sets=[lgb_train, lgb_valid],
                valid_names=['train', 'valid'],
                callbacks=callbacks
            )
        else:
            self.model = lgb.train(
                params,
                lgb_train,
                valid_sets=[lgb_train],
                valid_names=['train'],
                callbacks=callbacks
            )

    def predict(self, X_test):
        X_test_df = pd.DataFrame(X_test, columns=FEATURE_NAMES)

        # 填充所有缺失列为0
        if self.all_missing_cols:
            X_test_df[self.all_missing_cols] = 0

        # 对所有特征进行均值填充
        X_test_imputed = self.imputer.transform(X_test_df)

        # 标准化数据
        X_test_scaled = self.scaler.transform(X_test_imputed)

        # 预测
        y_pred_prob = self.model.predict(X_test_scaled, num_iteration=self.model.best_iteration)
        y_pred = y_pred_prob  # 对于后续信号分析，可以使用概率值
        return y_pred

    def hf_signal_test(self, X_test, y_test, dates, threshold=0.2):
        """
        测试高频信号
        """
        if self.model is None:
            raise ValueError("Model hasn't been trained yet")
        X_test_df = pd.DataFrame(X_test, columns=FEATURE_NAMES)

        # 填充所有缺失列为0
        if self.all_missing_cols:
            X_test_df[self.all_missing_cols] = 0

        # 对所有特征进行均值填充
        X_test_imputed = self.imputer.transform(X_test_df)

        # 标准化数据
        X_test_scaled = self.scaler.transform(X_test_imputed)

        # 转换为 DataFrame，便于处理 date_id
        y_test_df = pd.DataFrame({'label': y_test}, index=X_test_df.index)
        y_test_df['date_id'] = dates

        # 按照 date_id 进行去均值
        y_test_df['label'] = y_test_df['label'] - y_test_df.groupby('date_id')['label'].transform('mean')

        # 预测概率
        y_pred_prob = self.model.predict(X_test_scaled, num_iteration=self.model.best_iteration)
        y_test_df['pred'] = y_pred_prob

        # 计算信号指标
        up_precision_list = []
        down_precision_list = []
        up_alpha_list = []
        down_alpha_list = []

        for date in y_test_df['date_id'].unique():
            df_date = y_test_df[y_test_df['date_id'] == date]
            df_date = df_date.sort_values('pred')

            if int(threshold * len(df_date)) < 10:
                warnings.warn("Threshold is too low or not enough instruments on date {}".format(date))
                continue

            top = df_date.head(int(threshold * len(df_date)))
            bottom = df_date.tail(int(threshold * len(df_date)))

            down_precision = len(top[top['label'] < 0]) / len(top)
            up_precision = len(bottom[bottom['label'] > 0]) / len(bottom)

            down_alpha = top['label'].mean()
            up_alpha = bottom['label'].mean()

            up_precision_list.append(up_precision)
            down_precision_list.append(down_precision)
            up_alpha_list.append(up_alpha)
            down_alpha_list.append(down_alpha)

        print("===============================")
        print("High frequency signal test")
        print("===============================")
        print("Test set precision: ")
        print("Positive precision: {:.4f}, Negative precision: {:.4f}".format(
            np.mean(up_precision_list), np.mean(down_precision_list)))
        print("Test Alpha Average in test set: ")
        print("Positive average alpha: {:.6f}, Negative average alpha: {:.6f}".format(
            np.mean(up_alpha_list), np.mean(down_alpha_list)))

# ----------------- 模型训练函数 -----------------
def train(model_dict, model_name='lgb'):
    for i in range(N_FOLD):
        if TRAINING:
            selected_dates = [date for ii, date in enumerate(train_dates) if ii % N_FOLD != i]
            X_train = df[FEATURE_NAMES[:-1]].loc[df['date_id'].isin(selected_dates)].values
            y_train = df['responder_6'].loc[df['date_id'].isin(selected_dates)].values
            w_train = df['weight'].loc[df['date_id'].isin(selected_dates)].values
            symbol_id_train = df['symbol_id'].loc[df['date_id'].isin(selected_dates)].values

            if NUM_VALID_DATES > 0:
                X_valid = df[FEATURE_NAMES[:-1]].loc[df['date_id'].isin(valid_dates)].values
                y_valid = df['responder_6'].loc[df['date_id'].isin(valid_dates)].values
                w_valid = df['weight'].loc[df['date_id'].isin(valid_dates)].values
                symbol_id_valid = df['symbol_id'].loc[df['date_id'].isin(valid_dates)].values
            else:
                X_valid, y_valid, w_valid, symbol_id_valid = None, None, None, None

            # 将 symbol_id 作为特征添加到输入数据中
            X_train = np.hstack((X_train, symbol_id_train.reshape(-1, 1)))
            if X_valid is not None:
                X_valid = np.hstack((X_valid, symbol_id_valid.reshape(-1, 1)))

            model = model_dict[model_name]

            if model_name == 'hflgb':
                # HFLGBModel 训练过程
                model.fit(X_train, y_train, X_valid, y_valid, sample_weight=w_train, eval_sample_weight=w_valid)
            else:
                model.fit(
                    X_train, y_train, sample_weight=w_train,
                    eval_set=[(X_valid, y_valid)] if NUM_VALID_DATES > 0 else None,
                    early_stopping_rounds=200, verbose=10
                )

            # 保存模型
            joblib.dump(model, os.path.join(MODEL_DIR, f'{model_name}_{i}.model'))
            del X_train, y_train, w_train, X_valid, y_valid, w_valid, symbol_id_train, symbol_id_valid
        else:
            models.append(joblib.load(os.path.join(MODEL_PATH, f'{model_name}_{i}.model')))

# 收集模型的各折预测
def get_fold_predictions(model_names, df, dates, feature_names):
    fold_predictions = {model_name: [] for model_name in model_names}

    for model_name in model_names:
        for i in range(N_FOLD):
            model_path = os.path.join(MODEL_DIR, f'{model_name}_{i}.model')
            model = joblib.load(model_path)
            X = df[feature_names[:-1]].loc[df['date_id'].isin(dates)].values
            symbol_id = df['symbol_id'].loc[df['date_id'].isin(dates)].values

            # 将 symbol_id 作为特征添加到输入数据中
            X = np.hstack((X, symbol_id.reshape(-1, 1)))
            predictions = model.predict(X)
            fold_predictions[model_name].append(predictions)

    return fold_predictions

# 计算加权平均的预测
def ensemble_predictions(weights, fold_predictions):
    y_pred = np.zeros_like(fold_predictions[next(iter(fold_predictions))][0])
    for idx, model_name in enumerate(fold_predictions):
        avg_pred = np.mean(fold_predictions[model_name], axis=0)
        y_pred += weights[idx] * avg_pred
    return y_pred

# 优化权重
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

# ----------------- 模型字典 -----------------
model_dict = {
    'hflgb': HFLGBModel(
        n_estimators=5000,
        learning_rate=0.05,
        max_depth=6,
        device='gpu',
        gpu_use_dp=True,
        objective='binary',
        verbosity=-1
    ),
    'lgb': lgb.LGBMRegressor(
        n_estimators=5000,
        device='gpu',
        gpu_use_dp=True,
        objective='l2'
    ),
    'xgb': xgb.XGBRegressor(
        n_estimators=5000,
        learning_rate=0.1,
        max_depth=6,
        tree_method='hist',
        gpu_id=0,
        objective='reg:squarederror'
    ),
    'cbt': cbt.CatBoostRegressor(
        iterations=5000,
        learning_rate=0.05,
        task_type='GPU',
        loss_function='RMSE'
    ),
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
    dates_test = test_df['date_id'].values
    symbol_id_test = test_df['symbol_id'].values

    # 将 symbol_id 作为特征添加到输入数据中
    X_test = np.hstack((X_test, symbol_id_test.reshape(-1, 1)))

    model_names = list(model_dict.keys())
    fold_predictions = get_fold_predictions(model_names, df, test_dates, FEATURE_NAMES)

    optimized_weights = optimize_weights(fold_predictions, y_test, w_test)

    # 计算各个模型的分数
    model_scores = {}
    for model_name in model_names:
        avg_pred = np.mean(fold_predictions[model_name], axis=0)
        model_scores[model_name] = weighted_r2_score(y_test, avg_pred, w_test)
    print(f"最优模型权重: {dict(zip(model_names, optimized_weights))}")

    y_ensemble_pred = ensemble_predictions(optimized_weights, fold_predictions)
    ensemble_r2_score = weighted_r2_score(y_test, y_ensemble_pred, w_test)
    print(f"Ensemble 的加权 R² 分数: {ensemble_r2_score:.8f}")

    # 进行高频信号测试
    hflgb_model = model_dict['hflgb']
    hflgb_model.hf_signal_test(X_test, y_test, dates_test, threshold=0.2)
