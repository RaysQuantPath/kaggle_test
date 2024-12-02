import os
import joblib
import numpy as np
import pandas as pd
from scipy.optimize import minimize
import xgboost as xgb
import catboost as cbt
import lightgbm as lgb
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
import copy
import math
import warnings

# 忽略某些警告
warnings.filterwarnings("ignore")

# 文件路径和参数
ROOT_DIR = r'C:\Users\cyg19\Desktop\kaggle_test'
TRAIN_PATH = os.path.join(ROOT_DIR, 'filtered_train.parquet')
MODEL_DIR = './models_v11'
MODEL_PATH = './pretrained_models'

os.makedirs(ROOT_DIR, exist_ok=True)
os.makedirs(MODEL_DIR, exist_ok=True)
os.makedirs(MODEL_PATH, exist_ok=True)

# 全局常量
TRAINING = True
FEATURE_NAMES = [f"feature_{i:02d}" for i in range(79)]
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

# ----------------- XLSTM 模型 -----------------
class sLSTMCell(nn.Module):
    def __init__(self, input_size, hidden_size):
        super(sLSTMCell, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size

        # 定义参数
        self.w_i = nn.Parameter(torch.Tensor(hidden_size, input_size))
        self.w_f = nn.Parameter(torch.Tensor(hidden_size, input_size))
        self.w_o = nn.Parameter(torch.Tensor(hidden_size, input_size))
        self.w_z = nn.Parameter(torch.Tensor(hidden_size, input_size))

        self.r_i = nn.Parameter(torch.Tensor(hidden_size, hidden_size))
        self.r_f = nn.Parameter(torch.Tensor(hidden_size, hidden_size))
        self.r_o = nn.Parameter(torch.Tensor(hidden_size, hidden_size))
        self.r_z = nn.Parameter(torch.Tensor(hidden_size, hidden_size))

        self.b_i = nn.Parameter(torch.Tensor(hidden_size))
        self.b_f = nn.Parameter(torch.Tensor(hidden_size))
        self.b_o = nn.Parameter(torch.Tensor(hidden_size))
        self.b_z = nn.Parameter(torch.Tensor(hidden_size))

        self.sigmoid = nn.Sigmoid()

        self.reset_parameters()

    def reset_parameters(self):
        nn.init.xavier_uniform_(self.w_i)
        nn.init.xavier_uniform_(self.w_f)
        nn.init.xavier_uniform_(self.w_o)
        nn.init.xavier_uniform_(self.w_z)

        nn.init.orthogonal_(self.r_i)
        nn.init.orthogonal_(self.r_f)
        nn.init.orthogonal_(self.r_o)
        nn.init.orthogonal_(self.r_z)

        nn.init.zeros_(self.b_i)
        nn.init.zeros_(self.b_f)
        nn.init.zeros_(self.b_o)
        nn.init.zeros_(self.b_z)

    def forward(self, x, states):
        h_prev, c_prev, n_prev, m_prev = states
        h_prev = h_prev.to(x.device)
        c_prev = c_prev.to(x.device)
        n_prev = n_prev.to(x.device)
        m_prev = m_prev.to(x.device)

        # 添加数值稳定性处理
        epsilon = 1e-7

        i_tilda = (
            torch.matmul(self.w_i, x.transpose(0,1)).transpose(0,1)
            + torch.matmul(self.r_i, h_prev.transpose(0,1)).transpose(0,1)
            + self.b_i
        )
        f_tilda = (
            torch.matmul(self.w_f, x.transpose(0,1)).transpose(0,1)
            + torch.matmul(self.r_f, h_prev.transpose(0,1)).transpose(0,1)
            + self.b_f
        )
        o_tilda = (
            torch.matmul(self.w_o, x.transpose(0,1)).transpose(0,1)
            + torch.matmul(self.r_o, h_prev.transpose(0,1)).transpose(0,1)
            + self.b_o
        )
        z_tilda = (
            torch.matmul(self.w_z, x.transpose(0,1)).transpose(0,1)
            + torch.matmul(self.r_z, h_prev.transpose(0,1)).transpose(0,1)
            + self.b_z
        )

        # 防止指数溢出
        i_tilda = torch.clamp(i_tilda, -50, 50)
        f_tilda = torch.clamp(f_tilda, -50, 50)
        o_tilda = torch.clamp(o_tilda, -50, 50)
        z_tilda = torch.clamp(z_tilda, -50, 50)

        i_t = torch.exp(i_tilda)
        f_t = self.sigmoid(f_tilda)
        f_t = torch.clamp(f_t, min=epsilon, max=1 - epsilon)  # 避免 log(0)

        # 稳定器状态更新
        m_t = torch.max(torch.log(f_t) + m_prev, torch.log(i_t + epsilon))

        # 稳定化门
        i_prime = torch.exp(torch.log(i_t) - m_t)
        f_prime = torch.exp(torch.log(f_t) + m_prev - m_t)

        c_t = f_prime * c_prev + i_prime * torch.tanh(z_tilda)
        n_t = f_prime * n_prev + i_prime

        c_hat = c_t / (n_t + epsilon)  # 避免除以零
        h_t = self.sigmoid(o_tilda) * torch.tanh(c_hat)

        return h_t, (h_t, c_t, n_t, m_t)

class sLSTM(nn.Module):
    def __init__(self, input_size, seq_len, hidden_size, num_layers):
        super(sLSTM, self).__init__()
        self.layers = nn.ModuleList(
            [
                sLSTMCell(
                    input_size if i == 0 else hidden_size, hidden_size
                )
                for i in range(num_layers)
            ]
        )
        self.linear = nn.Linear(hidden_size, 1)
        self.outproj = nn.Linear(seq_len, 1)

        self.seq_len = seq_len
        self.input_size = input_size

    def forward(self, x, initial_states=None):
        batch_size, seq_len, _ = x.size()
        if initial_states is None:
            initial_states = [
                (
                    torch.zeros(
                        batch_size, self.layers[0].hidden_size
                    ).to(x.device),
                    torch.zeros(
                        batch_size, self.layers[0].hidden_size
                    ).to(x.device),
                    torch.zeros(
                        batch_size, self.layers[0].hidden_size
                    ).to(x.device),
                    torch.zeros(
                        batch_size, self.layers[0].hidden_size
                    ).to(x.device),
                )
                for _ in self.layers
            ]

        outputs = []
        current_states = initial_states

        for t in range(seq_len):
            x_t = x[:, t, :]
            new_states = []
            for layer, state in zip(self.layers, current_states):
                h_t, new_state = layer(x_t, state)
                new_states.append(new_state)
                x_t = h_t  # 传递到下一层
            outputs.append(h_t.unsqueeze(1))
            current_states = new_states

        outputs = torch.cat(
            outputs, dim=1
        )
        outputs = self.linear(outputs)
        outputs = outputs.squeeze(2)
        outputs = self.outproj(outputs)

        return outputs

class XLSTMWrapper:
    def __init__(self, input_size, seq_len=1, hidden_size=64, num_layers=2, dropout=0.0, batch_size=32, lr=0.0001, epochs=100, patience=5, device='cuda'):
        self.device = torch.device(device if torch.cuda.is_available() else 'cpu')
        self.model = sLSTM(input_size=input_size, seq_len=seq_len, hidden_size=hidden_size, num_layers=num_layers).to(self.device)
        self.batch_size = batch_size
        self.lr = lr
        self.epochs = epochs
        self.patience = patience  # 早停的耐心值
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=lr)
        self.loss_fn = torch.nn.MSELoss(reduction='none')  # 使用 'none' 以应用样本权重
        self.seq_len = seq_len
        self.input_size = input_size

        # 添加数据预处理器
        self.imputer = SimpleImputer(strategy='mean')
        self.scaler = StandardScaler()
        self.all_missing_cols = []  # 存储所有缺失的列

    def fit(self, X_train, y_train, X_valid=None, y_valid=None, sample_weight=None):
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

        # 填充所有缺失列为0（再次确保）
        if self.all_missing_cols:
            X_train_imputed[:, [FEATURE_NAMES.index(col) for col in self.all_missing_cols]] = 0
            if X_valid_imputed is not None:
                X_valid_imputed[:, [FEATURE_NAMES.index(col) for col in self.all_missing_cols]] = 0

        # 标准化数据
        X_train_scaled = self.scaler.fit_transform(X_train_imputed)
        if X_valid_imputed is not None:
            X_valid_scaled = self.scaler.transform(X_valid_imputed)
        else:
            X_valid_scaled = None

        # 重塑数据
        try:
            X_train_reshaped = X_train_scaled.reshape(-1, self.seq_len, self.input_size)
        except ValueError as e:
            print(f"Error reshaping X_train: {e}")
            print(f"Original shape: {X_train_scaled.shape}, Desired shape: (-1, {self.seq_len}, {self.input_size})")
            raise

        y_train = y_train.reshape(-1)

        if X_valid_scaled is not None:
            try:
                X_valid_reshaped = X_valid_scaled.reshape(-1, self.seq_len, self.input_size)
            except ValueError as e:
                print(f"Error reshaping X_valid: {e}")
                print(f"Original shape: {X_valid_scaled.shape}, Desired shape: (-1, {self.seq_len}, {self.input_size})")
                raise
            y_valid = y_valid.reshape(-1)
        else:
            X_valid_reshaped, y_valid = None, None

        # 转换为 PyTorch 张量
        dataset = TensorDataset(
            torch.tensor(X_train_reshaped, dtype=torch.float32),
            torch.tensor(y_train, dtype=torch.float32),
            torch.tensor(sample_weight if sample_weight is not None else np.ones(len(y_train)), dtype=torch.float32)
        )
        dataloader = DataLoader(dataset, batch_size=self.batch_size, shuffle=True)

        self.model.train()
        best_loss = float('inf')
        no_improve_epochs = 0  # 用于记录验证集上没有改进的 epoch 数量
        best_param = copy.deepcopy(self.model.state_dict())

        for epoch in range(self.epochs):
            epoch_loss = 0.0
            for X_batch, y_batch, w_batch in dataloader:
                X_batch, y_batch, w_batch = X_batch.to(self.device), y_batch.to(self.device), w_batch.to(self.device)
                self.optimizer.zero_grad()

                # 前向传播
                predictions = self.model(X_batch).squeeze(-1)

                # 计算加权损失
                loss = self.loss_fn(predictions, y_batch)
                weighted_loss = (loss * w_batch).mean()  # 应用样本权重

                # 反向传播和优化
                weighted_loss.backward()
                # 添加梯度裁剪
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
                self.optimizer.step()

                epoch_loss += weighted_loss.item()

            print(f"XLSTM Epoch {epoch + 1}/{self.epochs}, Training Loss: {epoch_loss:.4f}")

            # 验证阶段（如果提供验证数据）
            if X_valid_reshaped is not None and y_valid is not None:
                self.model.eval()
                with torch.no_grad():
                    X_valid_tensor = torch.tensor(X_valid_reshaped, dtype=torch.float32).to(self.device)
                    y_valid_tensor = torch.tensor(y_valid, dtype=torch.float32).to(self.device)
                    valid_predictions = self.model(X_valid_tensor).squeeze(-1)
                    valid_loss = torch.nn.functional.mse_loss(valid_predictions, y_valid_tensor, reduction='mean').item()

                print(f"XLSTM Epoch {epoch + 1}/{self.epochs}, Validation Loss: {valid_loss:.4f}")

                # 早停检查
                if valid_loss < best_loss:
                    best_loss = valid_loss
                    no_improve_epochs = 0
                    best_param = copy.deepcopy(self.model.state_dict())
                else:
                    no_improve_epochs += 1

                if no_improve_epochs >= self.patience:
                    print(f"Early stopping at epoch {epoch + 1}, best validation loss: {best_loss:.4f}")
                    break

                self.model.train()  # 恢复训练模式

        # 加载最佳参数
        if X_valid_reshaped is not None and y_valid is not None:
            self.model.load_state_dict(best_param)
            torch.save(best_param, os.path.join(MODEL_DIR, 'xlstm_best_param.pth'))

        if self.device.type == 'cuda':
            torch.cuda.empty_cache()

    def predict(self, X_test):
        # 转换为 DataFrame 以便处理
        X_test_df = pd.DataFrame(X_test, columns=FEATURE_NAMES)

        # 填充所有缺失列为0
        if self.all_missing_cols:
            X_test_df[self.all_missing_cols] = 0

        # 对所有特征进行均值填充
        X_test_imputed = self.imputer.transform(X_test_df)

        # 填充所有缺失列为0（再次确保）
        if self.all_missing_cols:
            X_test_imputed[:, [FEATURE_NAMES.index(col) for col in self.all_missing_cols]] = 0

        # 标准化数据
        X_test_scaled = self.scaler.transform(X_test_imputed)

        # 重塑数据
        try:
            X_test_reshaped = X_test_scaled.reshape(-1, self.seq_len, self.input_size)
        except ValueError as e:
            print(f"Error reshaping X_test: {e}")
            print(f"Original shape: {X_test_scaled.shape}, Desired shape: (-1, {self.seq_len}, {self.input_size})")
            raise

        self.model.eval()
        X_test_tensor = torch.tensor(X_test_reshaped, dtype=torch.float32).to(self.device)
        with torch.no_grad():
            predictions = self.model(X_test_tensor).cpu().numpy().squeeze(-1)
        return predictions

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
            if model_name in ['lgb', 'cbt', 'xgb']:
                if model_name == 'lgb':
                    model.fit(
                        X_train, y_train, sample_weight=w_train,
                        eval_set=[(X_valid, y_valid)] if NUM_VALID_DATES > 0 else None,
                        callbacks=[lgb.early_stopping(100), lgb.log_evaluation(10)] if NUM_VALID_DATES > 0 else None
                    )
                elif model_name == 'cbt':
                    if NUM_VALID_DATES > 0:
                        evalset = cbt.Pool(X_valid, y_valid, weight=w_valid)
                        model.fit(
                            X_train, y_train, sample_weight=w_train,
                            eval_set=[evalset], early_stopping_rounds=100, verbose=10
                        )
                    else:
                        model.fit(X_train, y_train, sample_weight=w_train)
                elif model_name == 'xgb':
                    model.fit(
                        X_train, y_train, sample_weight=w_train,
                        eval_set=[(X_valid, y_valid)] if NUM_VALID_DATES > 0 else None,
                        sample_weight_eval_set=[w_valid] if NUM_VALID_DATES > 0 else None,
                        early_stopping_rounds=100, verbose=10
                    )
            elif model_name == 'xlstm':
                # XLSTM 模型训练过程
                model.fit(X_train, y_train, X_valid, y_valid, sample_weight=w_train)

            # 保存模型
            joblib.dump(model, os.path.join(MODEL_DIR, f'{model_name}_{i}.model'))
            del X_train, y_train, w_train, X_valid, y_valid, w_valid
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
            if model_name == 'xlstm':
                # XLSTM 模型需要填充和标准化
                X_df = pd.DataFrame(X, columns=feature_names)
                if model.all_missing_cols:
                    X_df[model.all_missing_cols] = 0
                X_imputed = model.imputer.transform(X_df)
                if model.all_missing_cols:
                    X_imputed[:, [feature_names.index(col) for col in model.all_missing_cols]] = 0
                X_scaled = model.scaler.transform(X_imputed)
                try:
                    X_reshaped = X_scaled.reshape(-1, model.seq_len, model.input_size)
                except ValueError as e:
                    print(f"Error reshaping X for model {model_name}_{i}: {e}")
                    print(f"Original shape: {X_scaled.shape}, Desired shape: (-1, {model.seq_len}, {model.input_size})")
                    raise
                with torch.no_grad():
                    predictions = model.predict(X_scaled)  # 使用修正后的 predict 方法
                fold_predictions[model_name].append(predictions)
            else:
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
    'xlstm': XLSTMWrapper(input_size=len(FEATURE_NAMES), seq_len=1, hidden_size=64, num_layers=2, batch_size=32, lr=0.0001, epochs=100, patience=5, device='cuda'),
    'lgb': lgb.LGBMRegressor(n_estimators=500, device='gpu', gpu_use_dp=True, objective='l2'),
    'xgb': xgb.XGBRegressor(n_estimators=2000, learning_rate=0.1, max_depth=6, tree_method='hist', gpu_id=0,
                            objective='reg:squarederror'),
    'cbt': cbt.CatBoostRegressor(iterations=1000, learning_rate=0.05, task_type='GPU', loss_function='RMSE'),
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
