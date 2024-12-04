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
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.nn.modules.container import ModuleList
import math
import logging

# 设置日志
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# 忽略某些警告
warnings.filterwarnings("ignore")

# 文件路径和参数
ROOT_DIR = r'C:\Users\cyg19\Desktop\kaggle_test'
TRAIN_PATH = os.path.join(ROOT_DIR, 'filtered_train.parquet')
MODEL_DIR = './models_v16_v2'
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

        # 新增：处理 symbol_id
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

# ----------------- Localformer 模型 -----------------
class LocalformerModel:
    def __init__(
        self,
        d_feat=80,  # 特征数量，更新为包含 'symbol_id'
        d_model=64,
        batch_size=8192,
        nhead=2,
        num_layers=2,
        dropout=0,
        n_epochs=100,
        lr=0.0001,
        early_stop=5,
        loss="mse",
        optimizer="adam",
        reg=1e-3,
        n_jobs=10,
        device='cuda',
        seed=None,
    ):
        # 设置超参数
        self.d_feat = d_feat
        self.d_model = d_model
        self.batch_size = batch_size
        self.nhead = nhead
        self.num_layers = num_layers
        self.dropout = dropout
        self.n_epochs = n_epochs
        self.lr = lr
        self.early_stop = early_stop
        self.loss = loss
        self.optimizer_name = optimizer.lower()
        self.reg = reg
        self.n_jobs = n_jobs
        self.device = torch.device(device if torch.cuda.is_available() else "cpu")
        self.seed = seed
        self.logger = logger
        self.logger.info(
            f"LocalformerModel initialized with device {self.device}, batch_size {self.batch_size}"
        )
        if self.seed is not None:
            np.random.seed(self.seed)
            torch.manual_seed(self.seed)
        # 初始化模型
        self.model = Transformer(self.d_feat, d_model, nhead, num_layers, dropout, self.device)
        if self.optimizer_name == "adam":
            self.train_optimizer = optim.Adam(self.model.parameters(), lr=self.lr, weight_decay=self.reg)
        elif self.optimizer_name == "sgd":
            self.train_optimizer = optim.SGD(self.model.parameters(), lr=self.lr, weight_decay=self.reg)
        else:
            raise NotImplementedError(f"optimizer {self.optimizer_name} is not supported!")
        self.fitted = False
        self.model.to(self.device)

    def loss_fn(self, pred, label, sample_weight):
        loss = (pred - label) ** 2 * sample_weight
        return torch.mean(loss)

    def train_epoch(self, data_loader):
        self.model.train()
        for X_batch, y_batch, w_batch in data_loader:
            X_batch = X_batch.to(self.device)  # Shape [N, 1, F]
            y_batch = y_batch.to(self.device)
            w_batch = w_batch.to(self.device)
            self.train_optimizer.zero_grad()
            pred = self.model(X_batch)  # Output shape [N]
            loss = self.loss_fn(pred, y_batch, w_batch)
            if torch.isnan(loss) or torch.isinf(loss):
                self.logger.warning("Loss is NaN or Inf. Skipping this batch.")
                continue
            loss.backward()
            torch.nn.utils.clip_grad_value_(self.model.parameters(), 3.0)
            self.train_optimizer.step()

    def evaluate(self, data_loader):
        self.model.eval()
        total_loss = 0
        total_weight = 0
        with torch.no_grad():
            for X_batch, y_batch, w_batch in data_loader:
                X_batch = X_batch.to(self.device)
                y_batch = y_batch.to(self.device)
                w_batch = w_batch.to(self.device)
                pred = self.model(X_batch)
                loss = (pred - y_batch) ** 2 * w_batch
                total_loss += loss.sum().item()
                total_weight += w_batch.sum().item()
        if total_weight == 0:
            return float('inf')
        return total_loss / total_weight

    def fit(
        self,
        X_train,
        y_train,
        X_valid=None,
        y_valid=None,
        sample_weight=None,
        eval_sample_weight=None,
    ):
        # 处理 NaN 值，将其填充为 3
        X_train = np.nan_to_num(X_train, nan=3.0)
        y_train = np.nan_to_num(y_train, nan=3.0)
        if X_valid is not None and y_valid is not None:
            X_valid = np.nan_to_num(X_valid, nan=3.0)
            y_valid = np.nan_to_num(y_valid, nan=3.0)
        if sample_weight is not None:
            sample_weight = np.nan_to_num(sample_weight, nan=1.0)
        if eval_sample_weight is not None:
            eval_sample_weight = np.nan_to_num(eval_sample_weight, nan=1.0)
        # 转换数据为 PyTorch 张量
        self.logger.info("Preparing data...")
        X_train_tensor = torch.tensor(X_train, dtype=torch.float32).unsqueeze(1)  # Shape [N, 1, F]
        y_train_tensor = torch.tensor(y_train, dtype=torch.float32)
        if sample_weight is not None:
            w_train_tensor = torch.tensor(sample_weight, dtype=torch.float32)
        else:
            w_train_tensor = torch.ones_like(y_train_tensor)
        train_dataset = torch.utils.data.TensorDataset(X_train_tensor, y_train_tensor, w_train_tensor)
        # Windows 上将 num_workers 设置为 0，避免多进程问题
        num_workers = 0  # Windows 下使用 0
        train_loader = DataLoader(train_dataset, batch_size=self.batch_size, shuffle=True, num_workers=num_workers)
        if X_valid is not None and y_valid is not None:
            X_valid_tensor = torch.tensor(X_valid, dtype=torch.float32).unsqueeze(1)
            y_valid_tensor = torch.tensor(y_valid, dtype=torch.float32)
            if eval_sample_weight is not None:
                w_valid_tensor = torch.tensor(eval_sample_weight, dtype=torch.float32)
            else:
                w_valid_tensor = torch.ones_like(y_valid_tensor)
            valid_dataset = torch.utils.data.TensorDataset(X_valid_tensor, y_valid_tensor, w_valid_tensor)
            valid_loader = DataLoader(valid_dataset, batch_size=self.batch_size, shuffle=False, num_workers=num_workers)
        else:
            valid_loader = None

        best_loss = float('inf')
        best_model = None
        stop_steps = 0
        best_epoch = 0

        self.logger.info("Training...")
        for epoch in range(self.n_epochs):
            self.logger.info(f"Epoch {epoch}")
            self.train_epoch(train_loader)
            train_loss = self.evaluate(train_loader)
            if valid_loader is not None:
                val_loss = self.evaluate(valid_loader)
                self.logger.info(f"Train loss: {train_loss:.6f}, Validation loss: {val_loss:.6f}")
                if val_loss < best_loss:
                    best_loss = val_loss
                    best_model = copy.deepcopy(self.model.state_dict())
                    stop_steps = 0
                    best_epoch = epoch
                else:
                    stop_steps += 1
                    if stop_steps >= self.early_stop:
                        self.logger.info("Early stopping")
                        break
            else:
                self.logger.info(f"Train loss: {train_loss:.6f}")
        if best_model is not None:
            self.model.load_state_dict(best_model)
        self.fitted = True
        self.logger.info(f"Best validation loss: {best_loss:.6f} at epoch {best_epoch}")

    def predict(self, X_test):
        if not self.fitted:
            raise ValueError("Model is not fitted yet!")
        # 处理 NaN 值，将其填充为 3
        X_test = np.nan_to_num(X_test, nan=3.0)
        X_test_tensor = torch.tensor(X_test, dtype=torch.float32).unsqueeze(1)
        test_dataset = torch.utils.data.TensorDataset(X_test_tensor)
        num_workers = 0  # Windows 下使用 0
        test_loader = DataLoader(test_dataset, batch_size=self.batch_size, num_workers=num_workers)
        self.model.eval()
        preds = []
        with torch.no_grad():
            for X_batch in test_loader:
                X_batch = X_batch[0].to(self.device)
                pred = self.model(X_batch)
                preds.append(pred.cpu().numpy())
        return np.concatenate(preds)

    def save(self, path):
        torch.save(self.model.state_dict(), path)

    def load(self, path):
        self.model.load_state_dict(torch.load(path))
        self.fitted = True

class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=1000):
        super(PositionalEncoding, self).__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float32).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        if d_model % 2 == 1:
            pe[:, 1::2] = torch.cos(position * div_term[:-1])
        else:
            pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0).transpose(0, 1)
        self.register_buffer("pe", pe)

    def forward(self, x):
        # [T, N, F]
        return x + self.pe[: x.size(0), :]

def _get_clones(module, N):
    return ModuleList([copy.deepcopy(module) for i in range(N)])

class LocalformerEncoder(nn.Module):
    __constants__ = ["norm"]

    def __init__(self, encoder_layer, num_layers, d_model):
        super(LocalformerEncoder, self).__init__()
        self.layers = _get_clones(encoder_layer, num_layers)
        self.conv = _get_clones(nn.Conv1d(d_model, d_model, 3, 1, 1), num_layers)
        self.num_layers = num_layers

    def forward(self, src, mask):
        output = src
        out = src

        for i, mod in enumerate(self.layers):
            # [T, N, F] --> [N, T, F] --> [N, F, T]
            out = output.transpose(1, 0).transpose(2, 1)
            out = self.conv[i](out).transpose(2, 1).transpose(1, 0)

            output = mod(output + out, src_mask=mask)

        return output + out

class Transformer(nn.Module):
    def __init__(self, d_feat=80, d_model=64, nhead=4, num_layers=2, dropout=0.5, device=None):
        super(Transformer, self).__init__()
        self.rnn = nn.GRU(
            input_size=d_model,
            hidden_size=d_model,
            num_layers=num_layers,
            batch_first=False,
            dropout=dropout,
        )
        self.feature_layer = nn.Linear(d_feat, d_model)
        self.pos_encoder = PositionalEncoding(d_model)
        self.encoder_layer = nn.TransformerEncoderLayer(d_model=d_model, nhead=nhead, dropout=dropout)
        self.transformer_encoder = LocalformerEncoder(self.encoder_layer, num_layers=num_layers, d_model=d_model)
        self.decoder_layer = nn.Linear(d_model, 1)
        self.device = device
        self.d_feat = d_feat

    def forward(self, src):
        # src [N, T, F], [batch_size, seq_len, feature_dim]
        src = self.feature_layer(src)  # [batch_size, seq_len, d_model]

        # src [N, T, F] --> [T, N, F]
        src = src.transpose(1, 0)  # not batch first

        mask = None

        src = self.pos_encoder(src)
        output = self.transformer_encoder(src, mask)  # [seq_len, batch_size, d_model]

        output, _ = self.rnn(output)

        # [T, N, F] --> [N, T*F]
        output = self.decoder_layer(output.transpose(1, 0)[:, -1, :])  # [batch_size, 1]

        return output.squeeze()

# ----------------- 模型训练函数 -----------------
def train(model_dict, model_name='localformer'):
    for i in range(N_FOLD):
        if TRAINING:
            selected_dates = [date for ii, date in enumerate(train_dates) if ii % N_FOLD != i]
            X_train = df[FEATURE_NAMES[:-1]].loc[df['date_id'].isin(selected_dates)].values
            y_train = df['responder_6'].loc[df['date_id'].isin(selected_dates)].values
            w_train = df['weight'].loc[df['date_id'].isin(selected_dates)].values
            symbol_id_train = df['symbol_id'].loc[df['date_id'].isin(selected_dates)].values.reshape(-1, 1)

            if NUM_VALID_DATES > 0:
                X_valid = df[FEATURE_NAMES[:-1]].loc[df['date_id'].isin(valid_dates)].values
                y_valid = df['responder_6'].loc[df['date_id'].isin(valid_dates)].values
                w_valid = df['weight'].loc[df['date_id'].isin(valid_dates)].values
                symbol_id_valid = df['symbol_id'].loc[df['date_id'].isin(valid_dates)].values.reshape(-1, 1)
            else:
                X_valid, y_valid, w_valid, symbol_id_valid = None, None, None, None

            # 将 symbol_id 作为特征添加到输入数据中
            X_train = np.hstack((X_train, symbol_id_train))
            if X_valid is not None:
                X_valid = np.hstack((X_valid, symbol_id_valid))

            model = model_dict[model_name]
            if model_name == 'localformer':
                # LocalformerModel 训练过程
                model.fit(X_train, y_train, X_valid, y_valid, sample_weight=w_train, eval_sample_weight=w_valid)
                # 保存模型
                model.save(os.path.join(MODEL_DIR, f'{model_name}_{i}.model'))
            else:
                # 对于其他模型，将 symbol_id 作为特征
                model.fit(
                    X_train, y_train, sample_weight=w_train,
                    eval_set=[(X_valid, y_valid)] if NUM_VALID_DATES > 0 else None,
                    early_stopping_rounds=200, verbose=10
                )

            del X_train, y_train, w_train, X_valid, y_valid, w_valid, symbol_id_train, symbol_id_valid
        else:
            models.append(joblib.load(os.path.join(MODEL_PATH, f'{model_name}_{i}.model')))

# 收集模型的各折预测
def get_fold_predictions(model_names, df, dates, feature_names):
    fold_predictions = {model_name: [] for model_name in model_names}

    for model_name in model_names:
        for i in range(N_FOLD):
            model_path = os.path.join(MODEL_DIR, f'{model_name}_{i}.model')
            if model_name == 'localformer':
                model = LocalformerModel(
                    d_feat=len(feature_names),
                    n_epochs=100,
                    lr=0.0001,
                    batch_size=8192,
                    n_jobs=10,
                    device='cuda'
                )
                model.load(model_path)
            else:
                model = joblib.load(model_path)
            X = df[feature_names[:-1]].loc[df['date_id'].isin(dates)].values
            symbol_id = df['symbol_id'].loc[df['date_id'].isin(dates)].values.reshape(-1, 1)
            # 将 symbol_id 作为特征添加到输入数据中
            X = np.hstack((X, symbol_id))
            if model_name == 'localformer':
                predictions = model.predict(X)
                fold_predictions[model_name].append(predictions)
            else:
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

if __name__ == "__main__":
    # ----------------- 模型字典 -----------------
    model_dict = {
        'localformer': LocalformerModel(
            d_feat=len(FEATURE_NAMES),
            n_epochs=100,
            lr=0.0001,
            batch_size=8192,
            n_jobs=10,
            device='cuda'
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
            loss_function='RMSE'),
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
        symbol_id_test = test_df['symbol_id'].values.reshape(-1, 1)
        # 将 symbol_id 作为特征添加到输入数据中
        X_test = np.hstack((X_test, symbol_id_test))

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
