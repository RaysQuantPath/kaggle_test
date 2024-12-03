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

# 文件路径和参数
ROOT_DIR = r'C:\Users\cyg19\Desktop\kaggle_test'
TRAIN_PATH = os.path.join(ROOT_DIR, 'filtered_train.parquet')
MODEL_DIR = './models_v10'
MODEL_PATH = './pretrained_models'

os.makedirs(ROOT_DIR, exist_ok=True)
os.makedirs(MODEL_DIR, exist_ok=True)
os.makedirs(MODEL_PATH, exist_ok=True)

# 全局常量
TRAINING = True
FEATURE_NAMES = [f"feature_{i:02d}" for i in range(79)]
# NUM_VALID_DATES = 1
# NUM_TEST_DATES = 1
# SKIP_DATES = 1
# N_FOLD = 5
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

        print("已准备好训练、验证和测试数据集。")
    else:
        print(f"训练文件 '{TRAIN_PATH}' 为空。请提供有效的训练数据集。")
        exit()

# 定义加权 R² 评分函数
def weighted_r2_score(y_true, y_pred, weights):
    numerator = np.sum(weights * (y_true - y_pred) ** 2)
    denominator = np.sum(weights * (y_true - np.average(y_true, weights=weights)) ** 2)
    return 1 - (numerator / denominator)

# ----------------- LNN 模型 -----------------
class LTCRNN(torch.nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim=1, device='cuda:0'):
        super(LTCRNN, self).__init__()
        self.device = device
        self.rnn = torch.nn.RNN(input_dim, hidden_dim, batch_first=True)
        self.fc = torch.nn.Linear(hidden_dim, output_dim)

    def forward(self, x):
        _, hidden = self.rnn(x)
        return self.fc(hidden[-1])

class LNNWrapper:
    def __init__(self, input_dim, hidden_dim=64, batch_size=32, lr=0.01, epochs=100, patience=5, device="cuda"):
        self.model = LTCRNN(input_dim=input_dim, hidden_dim=hidden_dim, device=device).to(device)
        self.device = device
        self.batch_size = batch_size
        self.lr = lr
        self.epochs = epochs
        self.patience = patience  # 早停的耐心值
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=lr)
        self.loss_fn = torch.nn.MSELoss(reduction='none')  # Reduction set to 'none' to apply weights manually

    def fit(self, X_train, y_train, X_valid=None, y_valid=None, sample_weight=None):
        # Convert data to PyTorch tensors
        dataset = TensorDataset(
            torch.tensor(X_train, dtype=torch.float32),
            torch.tensor(y_train, dtype=torch.float32),
            torch.tensor(sample_weight if sample_weight is not None else np.ones(len(y_train)), dtype=torch.float32)
        )
        dataloader = DataLoader(dataset, batch_size=self.batch_size, shuffle=True)

        self.model.train()
        best_loss = float('inf')
        no_improve_epochs = 0  # 用于记录验证集上没有改进的 epoch 数量

        for epoch in range(self.epochs):
            # Training Phase
            epoch_loss = 0.0
            for X_batch, y_batch, w_batch in dataloader:
                X_batch, y_batch, w_batch = X_batch.to(self.device), y_batch.to(self.device), w_batch.to(self.device)
                self.optimizer.zero_grad()

                # Forward pass
                predictions = self.model(X_batch).squeeze(-1)

                # Compute weighted loss
                loss = self.loss_fn(predictions, y_batch)
                weighted_loss = (loss * w_batch).mean()  # Apply sample weights

                # Backward pass and optimizer step
                weighted_loss.backward()
                self.optimizer.step()

                epoch_loss += weighted_loss.item()

            print(f"LNN Epoch {epoch + 1}/{self.epochs}, Training Loss: {epoch_loss:.4f}")

            # Validation Phase (if validation data provided)
            if X_valid is not None and y_valid is not None:
                self.model.eval()
                X_valid_tensor = torch.tensor(X_valid, dtype=torch.float32).to(self.device)
                y_valid_tensor = torch.tensor(y_valid, dtype=torch.float32).to(self.device)
                with torch.no_grad():
                    valid_predictions = self.model(X_valid_tensor).squeeze(-1)
                    valid_loss = torch.nn.functional.mse_loss(valid_predictions, y_valid_tensor, reduction='mean').item()

                print(f"LNN Epoch {epoch + 1}/{self.epochs}, Validation Loss: {valid_loss:.4f}")

                # Early Stopping Check
                if valid_loss < best_loss:
                    best_loss = valid_loss
                    no_improve_epochs = 0
                else:
                    no_improve_epochs += 1

                if no_improve_epochs >= self.patience:
                    print(f"Early stopping at epoch {epoch + 1}, best validation loss: {best_loss:.4f}")
                    break

                self.model.train()  # Set back to train mode

    def predict(self, X_test):
        self.model.eval()
        X_test = torch.tensor(X_test, dtype=torch.float32).to(self.device)
        with torch.no_grad():
            return self.model(X_test).cpu().numpy()


# ----------------- 模型训练函数 -----------------
def train(model_dict, model_name='lgb'):
    for i in range(N_FOLD):
        if TRAINING:
            selected_dates = [date for ii, date in enumerate(train_dates) if ii % N_FOLD != i]
            X_train = df[FEATURE_NAMES].loc[df['date_id'].isin(selected_dates)].values
            y_train = df['responder_6'].loc[df['date_id'].isin(selected_dates)].values
            w_train = df['weight'].loc[df['date_id'].isin(selected_dates)].values

            if NUM_VALID_DATES > 0:
                X_valid = df[FEATURE_NAMES].loc[df['date_id'].isin(valid_dates)].values
                y_valid = df['responder_6'].loc[df['date_id'].isin(valid_dates)].values
                w_valid = df['weight'].loc[df['date_id'].isin(valid_dates)].values
            else:
                X_valid, y_valid, w_valid = None, None, None

            model = model_dict[model_name]
            if model_name == 'lgb':
                model.fit(
                    X_train, y_train, sample_weight=w_train,
                    eval_set=[(X_valid, y_valid)] if NUM_VALID_DATES > 0 else None,
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
            elif model_name == 'lnn':
                # LNN 模型训练过程
                X_train = np.nan_to_num(X_train, nan=3.0)
                y_train = np.nan_to_num(y_train, nan=3.0)
                model.fit(X_train, y_train, sample_weight=w_train)

            # 保存模型
            joblib.dump(model, os.path.join(MODEL_DIR, f'{model_name}_{i}.model'))
            del X_train, y_train, w_train
        else:
            models.append(joblib.load(os.path.join(MODEL_PATH, f'{model_name}_{i}.model')))


# 收集模型的各折预测
def get_fold_predictions(model_names, df, dates, feature_names):
    fold_predictions = {model_name: [] for model_name in model_names}

    for model_name in model_names:
        for i in range(N_FOLD):
            model_path = os.path.join(MODEL_DIR, f'{model_name}_{i}.model')
            model = joblib.load(model_path)
            X = df[feature_names].loc[df['date_id'].isin(dates)].values
            fold_predictions[model_name].append(model.predict(X))

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
    'lnn': LNNWrapper(input_dim=len(FEATURE_NAMES), hidden_dim=64, batch_size=32, lr=0.001, epochs=100),
    'lgb': lgb.LGBMRegressor(n_estimators=5000, device='gpu', gpu_use_dp=True, objective='l2'),
    'xgb': xgb.XGBRegressor(n_estimators=5000, learning_rate=0.1, max_depth=6, tree_method='hist', gpu_id=0,
                            objective='reg:squarederror'),
    'cbt': cbt.CatBoostRegressor(iterations=5000, learning_rate=0.05, task_type='GPU', loss_function='RMSE'),
}

# ----------------- 训练和测试 -----------------
models = []
for model_name in model_dict.keys():
    train(model_dict, model_name)

# 使用测试集评估
if TRAINING:
    test_df = df[df['date_id'].isin(test_dates)]
    X_test = test_df[FEATURE_NAMES].values
    y_test = test_df['responder_6'].values
    w_test = test_df['weight'].values

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
