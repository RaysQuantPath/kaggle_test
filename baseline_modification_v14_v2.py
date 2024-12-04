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
MODEL_DIR = './models_v14_v2'
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

        # 新增：处理 symbol_id
        symbol_ids = df['symbol_id'].unique()
        num_symbols = len(symbol_ids)
        symbol_id_to_index = {symbol_id: idx for idx, symbol_id in enumerate(symbol_ids)}

        print("已准备好训练、验证和测试数据集。")
    else:
        print(f"训练文件 '{TRAIN_PATH}' 为空。请提供有效的训练数据集。")
        exit()

# 定义加权 R² 评分函数
def weighted_r2_score(y_true, y_pred, weights):
    numerator = np.sum(weights * (y_true - y_pred) ** 2)
    denominator = np.sum(weights * (y_true - np.average(y_true, weights=weights)) ** 2)
    return 1 - (numerator / denominator)

# ----------------- SFM 模型 -----------------
class SFM_Model(nn.Module):
    def __init__(
        self,
        d_feat=79,
        output_dim=1,
        freq_dim=10,
        hidden_size=64,
        dropout_W=0.0,
        dropout_U=0.0,
        device="cpu",
        num_symbols=1,
    ):
        super().__init__()

        self.input_dim = d_feat
        self.output_dim = output_dim
        self.freq_dim = freq_dim
        self.hidden_dim = hidden_size
        self.device = device
        self.num_symbols = num_symbols

        self.W_i = nn.Parameter(torch.empty((self.input_dim, self.hidden_dim)))
        nn.init.xavier_uniform_(self.W_i)
        self.U_i = nn.Parameter(torch.empty(self.hidden_dim, self.hidden_dim))
        nn.init.orthogonal_(self.U_i)
        self.b_i = nn.Parameter(torch.zeros(self.hidden_dim))

        self.W_ste = nn.Parameter(torch.empty(self.input_dim, self.hidden_dim))
        nn.init.xavier_uniform_(self.W_ste)
        self.U_ste = nn.Parameter(torch.empty(self.hidden_dim, self.hidden_dim))
        nn.init.orthogonal_(self.U_ste)
        self.b_ste = nn.Parameter(torch.ones(self.hidden_dim))

        self.W_fre = nn.Parameter(torch.empty(self.input_dim, self.freq_dim))
        nn.init.xavier_uniform_(self.W_fre)
        self.U_fre = nn.Parameter(torch.empty(self.hidden_dim, self.freq_dim))
        nn.init.orthogonal_(self.U_fre)
        self.b_fre = nn.Parameter(torch.ones(self.freq_dim))

        self.W_c = nn.Parameter(torch.empty(self.input_dim, self.hidden_dim))
        nn.init.xavier_uniform_(self.W_c)
        self.U_c = nn.Parameter(torch.empty(self.hidden_dim, self.hidden_dim))
        nn.init.orthogonal_(self.U_c)
        self.b_c = nn.Parameter(torch.zeros(self.hidden_dim))

        self.W_o = nn.Parameter(torch.empty(self.input_dim, self.hidden_dim))
        nn.init.xavier_uniform_(self.W_o)
        self.U_o = nn.Parameter(torch.empty(self.hidden_dim, self.hidden_dim))
        nn.init.orthogonal_(self.U_o)
        self.b_o = nn.Parameter(torch.zeros(self.hidden_dim))

        self.U_a = nn.Parameter(torch.empty(self.freq_dim, 1))
        nn.init.orthogonal_(self.U_a)
        self.b_a = nn.Parameter(torch.zeros(self.hidden_dim))

        self.W_p = nn.Parameter(torch.empty(self.hidden_dim, self.output_dim * self.num_symbols))
        nn.init.xavier_uniform_(self.W_p)
        self.b_p = nn.Parameter(torch.zeros(self.output_dim * self.num_symbols))

        self.activation = nn.Tanh()
        self.inner_activation = nn.Hardsigmoid()
        self.dropout_W, self.dropout_U = (dropout_W, dropout_U)

        self.fc_out = nn.Linear(self.output_dim * self.num_symbols, self.num_symbols)

        self.states = []

    def forward(self, input):
        input = input.reshape(len(input), self.input_dim, -1)  # [N, F, T]
        input = input.permute(0, 2, 1)  # [N, T, F]
        time_step = input.shape[1]

        for ts in range(time_step):
            x = input[:, ts, :]
            if len(self.states) == 0:  # hasn't initialized yet
                self.init_states(x)
            self.get_constants(x)
            p_tm1 = self.states[0]  # noqa: F841
            h_tm1 = self.states[1]
            S_re_tm1 = self.states[2]
            S_im_tm1 = self.states[3]
            time_tm1 = self.states[4]
            B_U = self.states[5]
            B_W = self.states[6]
            frequency = self.states[7]

            x_i = torch.matmul(x * B_W[0], self.W_i) + self.b_i
            x_ste = torch.matmul(x * B_W[0], self.W_ste) + self.b_ste
            x_fre = torch.matmul(x * B_W[0], self.W_fre) + self.b_fre
            x_c = torch.matmul(x * B_W[0], self.W_c) + self.b_c
            x_o = torch.matmul(x * B_W[0], self.W_o) + self.b_o

            i = self.inner_activation(x_i + torch.matmul(h_tm1 * B_U[0], self.U_i))
            ste = self.inner_activation(x_ste + torch.matmul(h_tm1 * B_U[0], self.U_ste))
            fre = self.inner_activation(x_fre + torch.matmul(h_tm1 * B_U[0], self.U_fre))

            ste = torch.reshape(ste, (-1, self.hidden_dim, 1))
            fre = torch.reshape(fre, (-1, 1, self.freq_dim))

            f = ste * fre

            c = i * self.activation(x_c + torch.matmul(h_tm1 * B_U[0], self.U_c))

            time = time_tm1 + 1

            omega = torch.tensor(2 * np.pi).to(self.device) * time * frequency

            re = torch.cos(omega)
            im = torch.sin(omega)

            c = torch.reshape(c, (-1, self.hidden_dim, 1))

            S_re = f * S_re_tm1 + c * re
            S_im = f * S_im_tm1 + c * im

            A = torch.square(S_re) + torch.square(S_im)

            A = torch.reshape(A, (-1, self.freq_dim)).float()
            A_a = torch.matmul(A * B_U[0], self.U_a)
            A_a = torch.reshape(A_a, (-1, self.hidden_dim))
            a = self.activation(A_a + self.b_a)

            o = self.inner_activation(x_o + torch.matmul(h_tm1 * B_U[0], self.U_o))

            h = o * a
            p = torch.matmul(h, self.W_p) + self.b_p

            self.states = [p, h, S_re, S_im, time, None, None, None]
        self.states = []

        output = self.fc_out(p)  # 输出维度为 (batch_size, num_symbols)
        return output  # 返回所有 symbol 的预测

    def init_states(self, x):
        batch_size = x.size(0)
        init_state_h = torch.zeros(batch_size, self.hidden_dim).to(self.device)
        init_state_p = torch.zeros(batch_size, self.output_dim * self.num_symbols).to(self.device)

        init_state = torch.zeros(batch_size, self.hidden_dim).to(self.device)
        init_freq = torch.zeros(batch_size, self.freq_dim).to(self.device)

        init_state = torch.reshape(init_state, (-1, self.hidden_dim, 1))
        init_freq = torch.reshape(init_freq, (-1, 1, self.freq_dim))

        init_state_S_re = init_state * init_freq
        init_state_S_im = init_state * init_freq

        init_state_time = torch.tensor(0).to(self.device)

        self.states = [
            init_state_p,
            init_state_h,
            init_state_S_re,
            init_state_S_im,
            init_state_time,
            None,
            None,
            None,
        ]

    def get_constants(self, x):
        constants = []
        constants.append([torch.tensor(1.0).to(self.device) for _ in range(6)])
        constants.append([torch.tensor(1.0).to(self.device) for _ in range(7)])
        array = np.array([float(ii) / self.freq_dim for ii in range(self.freq_dim)])
        constants.append(torch.tensor(array).to(self.device))

        self.states[5:] = constants

class SFMWrapper:
    def __init__(
        self,
        input_size,
        output_dim=1,
        freq_dim=10,
        hidden_size=64,
        dropout_W=0.0,
        dropout_U=0.0,
        batch_size=32,
        lr=0.0001,
        epochs=100,
        patience=5,
        device='cuda',
        num_symbols=1,
    ):
        self.device = torch.device(device if torch.cuda.is_available() else 'cpu')
        self.model = SFM_Model(
            d_feat=input_size,
            output_dim=output_dim,
            freq_dim=freq_dim,
            hidden_size=hidden_size,
            dropout_W=dropout_W,
            dropout_U=dropout_U,
            device=self.device,
            num_symbols=num_symbols,
        ).to(self.device)
        self.batch_size = batch_size
        self.lr = lr
        self.epochs = epochs
        self.patience = patience  # 早停的耐心值
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=lr)
        self.loss_fn = torch.nn.MSELoss(reduction='none')  # 使用 'none' 以应用样本权重
        self.input_size = input_size

        # 添加数据预处理器
        self.imputer = SimpleImputer(strategy='mean')
        self.scaler = StandardScaler()
        self.all_missing_cols = []  # 存储所有缺失的列

    def __getstate__(self):
        state = self.__dict__.copy()
        state['device'] = state['device'].type
        return state

    def __setstate__(self, state):
        device_str = state.pop('device')
        state['device'] = torch.device(device_str)
        self.__dict__.update(state)

    def fit(self, X_train, y_train, symbol_id_train, X_valid=None, y_valid=None, symbol_id_valid=None, sample_weight=None):
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

        y_train = y_train.reshape(-1)

        if X_valid_scaled is not None:
            y_valid = y_valid.reshape(-1)
        else:
            y_valid = None

        # 将 symbol_id 转换为张量
        symbol_id_tensor = torch.tensor(symbol_id_train, dtype=torch.long)
        if symbol_id_valid is not None:
            symbol_id_valid_tensor = torch.tensor(symbol_id_valid, dtype=torch.long)

        # 转换为 PyTorch 张量
        dataset = TensorDataset(
            torch.tensor(X_train_scaled, dtype=torch.float32),
            torch.tensor(y_train, dtype=torch.float32),
            torch.tensor(sample_weight if sample_weight is not None else np.ones(len(y_train)), dtype=torch.float32),
            symbol_id_tensor
        )
        dataloader = DataLoader(dataset, batch_size=self.batch_size, shuffle=True)

        self.model.train()
        best_loss = float('inf')
        no_improve_epochs = 0  # 用于记录验证集上没有改进的 epoch 数量
        best_param = copy.deepcopy(self.model.state_dict())

        for epoch in range(self.epochs):
            epoch_loss = 0.0
            for X_batch, y_batch, w_batch, symbol_id_batch in dataloader:
                X_batch, y_batch, w_batch, symbol_id_batch = (
                    X_batch.to(self.device),
                    y_batch.to(self.device),
                    w_batch.to(self.device),
                    symbol_id_batch.to(self.device)
                )
                self.optimizer.zero_grad()

                # 前向传播
                predictions = self.model(X_batch)  # (batch_size, num_symbols)

                # 选择对应的 symbol_id 的预测
                predictions_selected = predictions.gather(1, symbol_id_batch.unsqueeze(1)).squeeze(1)

                # 计算加权损失
                loss = self.loss_fn(predictions_selected, y_batch)
                weighted_loss = (loss * w_batch).mean()  # 应用样本权重

                # 反向传播和优化
                weighted_loss.backward()
                # 添加梯度裁剪
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
                self.optimizer.step()

                epoch_loss += weighted_loss.item()

            print(f"SFM Epoch {epoch + 1}/{self.epochs}, Training Loss: {epoch_loss:.4f}")

            # 验证阶段（如果提供验证数据）
            if X_valid_scaled is not None and y_valid is not None:
                self.model.eval()
                with torch.no_grad():
                    X_valid_tensor = torch.tensor(X_valid_scaled, dtype=torch.float32).to(self.device)
                    y_valid_tensor = torch.tensor(y_valid, dtype=torch.float32).to(self.device)
                    symbol_id_valid_tensor = symbol_id_valid_tensor.to(self.device)
                    valid_predictions = self.model(X_valid_tensor)  # (batch_size, num_symbols)
                    valid_predictions_selected = valid_predictions.gather(1, symbol_id_valid_tensor.unsqueeze(1)).squeeze(1)
                    valid_loss = torch.nn.functional.mse_loss(valid_predictions_selected, y_valid_tensor, reduction='mean').item()

                print(f"SFM Epoch {epoch + 1}/{self.epochs}, Validation Loss: {valid_loss:.4f}")

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
        if X_valid_scaled is not None and y_valid is not None:
            self.model.load_state_dict(best_param)
            torch.save(best_param, os.path.join(MODEL_DIR, 'sfm_best_param.pth'))

        if self.device.type == 'cuda':
            torch.cuda.empty_cache()

    def predict(self, X_test, symbol_id_test):
        # 转换为 DataFrame 以便处理
        X_test_df = pd.DataFrame(X_test, columns=FEATURE_NAMES)

        # 填充所有缺失列为0
        if self.all_missing_cols:
            X_test_df[self.all_missing_cols] = 0

        # 对所有特征进行均值填充
        X_test_imputed = self.imputer.transform(X_test_df)

        # 标准化数据
        X_test_scaled = self.scaler.transform(X_test_imputed)

        self.model.eval()
        X_test_tensor = torch.tensor(X_test_scaled, dtype=torch.float32).to(self.device)
        symbol_id_tensor = torch.tensor([symbol_id_to_index[sid] for sid in symbol_id_test], dtype=torch.long).to(self.device)

        with torch.no_grad():
            predictions = self.model(X_test_tensor)  # (num_samples, num_symbols)
            predictions_selected = predictions.gather(1, symbol_id_tensor.unsqueeze(1)).cpu().numpy().squeeze(1)
        return predictions_selected

# ----------------- 模型训练函数 -----------------
def train(model_dict, model_name='lgb'):
    for i in range(N_FOLD):
        if TRAINING:
            selected_dates = [date for ii, date in enumerate(train_dates) if ii % N_FOLD != i]
            X_train = df[FEATURE_NAMES].loc[df['date_id'].isin(selected_dates)].values
            y_train = df['responder_6'].loc[df['date_id'].isin(selected_dates)].values
            w_train = df['weight'].loc[df['date_id'].isin(selected_dates)].values
            symbol_id_train = df['symbol_id'].loc[df['date_id'].isin(selected_dates)].values
            symbol_id_train = np.array([symbol_id_to_index[sid] for sid in symbol_id_train])

            if NUM_VALID_DATES > 0:
                X_valid = df[FEATURE_NAMES].loc[df['date_id'].isin(valid_dates)].values
                y_valid = df['responder_6'].loc[df['date_id'].isin(valid_dates)].values
                w_valid = df['weight'].loc[df['date_id'].isin(valid_dates)].values
                symbol_id_valid = df['symbol_id'].loc[df['date_id'].isin(valid_dates)].values
                symbol_id_valid = np.array([symbol_id_to_index[sid] for sid in symbol_id_valid])
            else:
                X_valid, y_valid, w_valid, symbol_id_valid = None, None, None, None

            model = model_dict[model_name]
            if model_name == 'sfm':
                # SFM 模型训练过程
                model.fit(X_train, y_train, symbol_id_train, X_valid, y_valid, symbol_id_valid, sample_weight=w_train)
            else:
                # 对于其他模型，将 symbol_id 作为特征
                X_train = np.hstack((X_train, symbol_id_train.reshape(-1, 1)))
                if X_valid is not None:
                    X_valid = np.hstack((X_valid, symbol_id_valid.reshape(-1, 1)))
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
            X = df[feature_names].loc[df['date_id'].isin(dates)].values
            symbol_id = df['symbol_id'].loc[df['date_id'].isin(dates)].values
            symbol_id = np.array([symbol_id_to_index[sid] for sid in symbol_id])

            if model_name == 'sfm':
                predictions = model.predict(X, symbol_id)
                fold_predictions[model_name].append(predictions)
            else:
                # 对于其他模型，将 symbol_id 作为特征
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
    'sfm': SFMWrapper(
        input_size=len(FEATURE_NAMES),
        output_dim=1,
        freq_dim=10,
        hidden_size=64,
        dropout_W=0.0,
        dropout_U=0.0,
        batch_size=32,
        lr=0.0001,
        epochs=100,
        patience=5,
        device='cuda',
        num_symbols=num_symbols
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
    X_test = test_df[FEATURE_NAMES].values
    y_test = test_df['responder_6'].values
    w_test = test_df['weight'].values
    symbol_id_test = test_df['symbol_id'].values
    symbol_id_test = np.array([symbol_id_to_index[sid] for sid in symbol_id_test])

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
