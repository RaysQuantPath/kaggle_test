import os
import joblib
import numpy as np
import pandas as pd
import xgboost as xgb
import catboost as cbt
import lightgbm as lgb
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
import copy
import warnings
import random
from deap import base, creator, tools, algorithms
from joblib import Parallel, delayed
import multiprocessing
import numba


# 忽略某些警告
warnings.filterwarnings("ignore")

# 文件路径和参数
ROOT_DIR = r'C:\Users\cyg19\Desktop\kaggle_test'
TRAIN_PATH = os.path.join(ROOT_DIR, 'filtered_train.parquet')
MODEL_DIR = './models_v11_v2'
MODEL_PATH = './pretrained_models'

os.makedirs(ROOT_DIR, exist_ok=True)
os.makedirs(MODEL_DIR, exist_ok=True)
os.makedirs(MODEL_PATH, exist_ok=True)

# 全局常量
TRAINING = True
FEATURE_NAMES = [f"feature_{i:02d}" for i in range(79)]
FEATURE_NAMES_XLSTM = [f"feature_{i:02d}" for i in range(79)]  # 仅用于 XLSTM
FEATURE_NAMES_OTHER = [f"feature_{i:02d}" for i in range(79)] + ['symbol_id']  # 其他模型包含 symbol_id
NUM_VALID_DATES = 100
NUM_TEST_DATES = 90
SKIP_DATES = 500
N_FOLD = 5
SEQUENCE_LENGTH = 5


def reduce_mem_usage(self, float16_as32=True):
    #memory_usage()是df每列的内存使用量,sum是对它们求和, B->KB->MB
    start_mem = df.memory_usage().sum() / 1024**2
    print('Memory usage of dataframe is {:.2f} MB'.format(start_mem))

    for col in df.columns:#遍历每列的列名
        col_type = df[col].dtype#列名的type
        if col_type != object and str(col_type)!='category':#不是object也就是说这里处理的是数值类型的变量
            c_min,c_max = df[col].min(),df[col].max() #求出这列的最大值和最小值
            if str(col_type)[:3] == 'int':#如果是int类型的变量,不管是int8,int16,int32还是int64
                #如果这列的取值范围是在int8的取值范围内,那就对类型进行转换 (-128 到 127)
                if c_min > np.iinfo(np.int8).min and c_max < np.iinfo(np.int8).max:
                    df[col] = df[col].astype(np.int8)
                #如果这列的取值范围是在int16的取值范围内,那就对类型进行转换(-32,768 到 32,767)
                elif c_min > np.iinfo(np.int16).min and c_max < np.iinfo(np.int16).max:
                    df[col] = df[col].astype(np.int16)
                #如果这列的取值范围是在int32的取值范围内,那就对类型进行转换(-2,147,483,648到2,147,483,647)
                elif c_min > np.iinfo(np.int32).min and c_max < np.iinfo(np.int32).max:
                    df[col] = df[col].astype(np.int32)
                #如果这列的取值范围是在int64的取值范围内,那就对类型进行转换(-9,223,372,036,854,775,808到9,223,372,036,854,775,807)
                elif c_min > np.iinfo(np.int64).min and c_max < np.iinfo(np.int64).max:
                    df[col] = df[col].astype(np.int64)
            else:#如果是浮点数类型.
                #如果数值在float16的取值范围内,如果觉得需要更高精度可以考虑float32
                if c_min > np.finfo(np.float16).min and c_max < np.finfo(np.float16).max:
                    if float16_as32:#如果数据需要更高的精度可以选择float32
                        df[col] = df[col].astype(np.float32)
                    else:
                        df[col] = df[col].astype(np.float16)
                #如果数值在float32的取值范围内，对它进行类型转换
                elif c_min > np.finfo(np.float32).min and c_max < np.finfo(np.float32).max:
                    df[col] = df[col].astype(np.float32)
                #如果数值在float64的取值范围内，对它进行类型转换
                else:
                    df[col] = df[col].astype(np.float64)
    #计算一下结束后的内存
    end_mem = df.memory_usage().sum() / 1024**2
    print('Memory usage after optimization is: {:.2f} MB'.format(end_mem))
    #相比一开始的内存减少了百分之多少
    print('Decreased by {:.1f}%'.format(100 * (start_mem - end_mem) / start_mem))

    return df

# 加载训练数据
if TRAINING:
    if os.path.getsize(TRAIN_PATH) > 0:
        df = pd.read_parquet(TRAIN_PATH)
        df = reduce_mem_usage(df, False)
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
        index_to_symbol_id = {idx: symbol_id for idx, symbol_id in enumerate(symbol_ids)}

        print("已准备好训练、验证和测试数据集。")
    else:
        print(f"训练文件 '{TRAIN_PATH}' 为空。请提供有效的训练数据集。")
        exit()


# 定义加权 R² 评分函数
def weighted_r2_score(y_true, y_pred, weights):
    numerator = np.sum(weights * (y_true - y_pred) ** 2)
    denominator = np.sum(weights * (y_true - np.average(y_true, weights=weights)) ** 2)
    return 1 - (numerator / denominator)


# 定义创建序列并前向填充的函数
def create_sequences_with_padding(df, feature_names, sequence_length, symbol_id_to_index):

    @numba.njit(parallel=True, fastmath=True)
    def build_sequences_numba(group_features, sequence_length):
        n, feature_num = group_features.shape
        out_len = n - sequence_length + 1
        sequences = np.empty((out_len, sequence_length, feature_num), dtype=group_features.dtype)
        for i in numba.prange(out_len):
            for j in range(sequence_length):
                sequences[i, j, :] = group_features[i + j]
        return sequences

    def process_group(symbol_id, group, feature_names, sequence_length, symbol_id_to_index):
        group_features = group[feature_names].values
        group_targets = group['responder_6'].values
        group_weights = group['weight'].values
        n = len(group_features)

        if n == 0:
            return [], [], [], []

        sequences = []
        targets = []
        weights = []
        symbol_ids_seq = []
        pad_count = sequence_length - 1
        feature_num = group_features.shape[1]

        # 构建pad_seq
        if pad_count > 0:
            pad_seq = np.tile(group_features[0].reshape(1, 1, feature_num), (pad_count, sequence_length, 1))
            pad_target = np.full(pad_count, group_targets[0])
            pad_weight = np.full(pad_count, group_weights[0])
            pad_symbol_id = np.full(pad_count, symbol_id_to_index[symbol_id])
            sequences.append(pad_seq)
            targets.append(pad_target)
            weights.append(pad_weight)
            symbol_ids_seq.append(pad_symbol_id)

        # 构建正常序列
        if n >= sequence_length:
            # 使用numba加速构造序列
            normal_seqs = build_sequences_numba(group_features, sequence_length)
            normal_targets = group_targets[sequence_length - 1:]
            normal_weights = group_weights[sequence_length - 1:]
            normal_symbol_id = np.full(n - sequence_length + 1, symbol_id_to_index[symbol_id])

            sequences.append(normal_seqs)
            targets.append(normal_targets)
            weights.append(normal_weights)
            symbol_ids_seq.append(normal_symbol_id)

        return sequences, targets, weights, symbol_ids_seq

    grouped = df.groupby('symbol_id')
    num_cores = multiprocessing.cpu_count() - 1

    results = Parallel(n_jobs=num_cores)(
        delayed(process_group)(symbol_id, group, feature_names, sequence_length, symbol_id_to_index)
        for symbol_id, group in grouped
    )

    all_sequences = []
    all_targets = []
    all_weights = []
    all_symbol_ids_seq = []

    for sequences, targets, weights, symbol_ids_seq in results:
        if sequences:
            all_sequences.append(np.concatenate(sequences, axis=0))
            all_targets.append(np.concatenate(targets, axis=0))
            all_weights.append(np.concatenate(weights, axis=0))
            all_symbol_ids_seq.append(np.concatenate(symbol_ids_seq, axis=0))

    if all_sequences:
        X = np.concatenate(all_sequences, axis=0)
        y = np.concatenate(all_targets, axis=0)
        w = np.concatenate(all_weights, axis=0)
        symbol_ids_seq = np.concatenate(all_symbol_ids_seq, axis=0)
    else:
        X = np.empty((0, sequence_length, len(feature_names)))
        y = np.empty((0,))
        w = np.empty((0,))
        symbol_ids_seq = np.empty((0,), dtype=int)

    return X, y, w, symbol_ids_seq


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
                torch.matmul(self.w_i, x.transpose(0, 1)).transpose(0, 1)
                + torch.matmul(self.r_i, h_prev.transpose(0, 1)).transpose(0, 1)
                + self.b_i
        )
        f_tilda = (
                torch.matmul(self.w_f, x.transpose(0, 1)).transpose(0, 1)
                + torch.matmul(self.r_f, h_prev.transpose(0, 1)).transpose(0, 1)
                + self.b_f
        )
        o_tilda = (
                torch.matmul(self.w_o, x.transpose(0, 1)).transpose(0, 1)
                + torch.matmul(self.r_o, h_prev.transpose(0, 1)).transpose(0, 1)
                + self.b_o
        )
        z_tilda = (
                torch.matmul(self.w_z, x.transpose(0, 1)).transpose(0, 1)
                + torch.matmul(self.r_z, h_prev.transpose(0, 1)).transpose(0, 1)
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
    def __init__(self, input_size, seq_len, hidden_size, num_layers, num_symbols):
        super(sLSTM, self).__init__()
        self.layers = nn.ModuleList(
            [
                sLSTMCell(
                    input_size if i == 0 else hidden_size, hidden_size
                )
                for i in range(num_layers)
            ]
        )
        self.linear = nn.Linear(hidden_size, num_symbols)
        self.seq_len = seq_len
        self.input_size = input_size
        self.num_symbols = num_symbols

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
        )  # (batch_size, seq_len, hidden_size)
        outputs = outputs[:, -1, :]  # 取最后一个时间步的输出

        outputs = self.linear(outputs)  # (batch_size, num_symbols)

        return outputs


class XLSTMWrapper:
    def __init__(self, input_size, seq_len=1, hidden_size=64, num_layers=2, num_symbols=1, dropout=0.0, batch_size=32,
                 lr=0.0001, epochs=100, patience=5, device='cuda'):
        self.device = torch.device(device if torch.cuda.is_available() else 'cpu')
        self.model = sLSTM(input_size=input_size, seq_len=seq_len, hidden_size=hidden_size, num_layers=num_layers,
                           num_symbols=num_symbols).to(self.device)
        self.batch_size = batch_size
        self.lr = lr
        self.epochs = epochs
        self.patience = patience  # 早停的耐心值
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=lr)
        self.loss_fn = torch.nn.MSELoss(reduction='none')  # 使用 'none' 以应用样本权重
        self.seq_len = seq_len
        self.input_size = input_size

        # 添加数据预处理器
        self.imputer = SimpleImputer(strategy='constant', fill_value=3)  # 将缺失值填充为 3
        self.scaler = StandardScaler()
        self.all_missing_cols = []  # 存储所有缺失的列

    def fit(self, X_train, y_train, symbol_id_train, X_valid=None, y_valid=None, symbol_id_valid=None,
            sample_weight=None):
        # 转换为 DataFrame 以便处理
        flat_feature_names = [f"feature_{i:02d}" for i in range(self.input_size)]
        X_train_flat = X_train.reshape(-1, self.input_size)
        X_train_df = pd.DataFrame(X_train_flat, columns=flat_feature_names)
        X_valid_flat = X_valid.reshape(-1, self.input_size) if X_valid is not None else None
        X_valid_df = pd.DataFrame(X_valid_flat, columns=flat_feature_names) if X_valid_flat is not None else None

        # 填充缺失值为 3
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

        # 重塑数据回原形状
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

        # 将 symbol_id 转换为张量
        symbol_id_tensor = torch.tensor(symbol_id_train, dtype=torch.long)
        if symbol_id_valid is not None:
            symbol_id_valid_tensor = torch.tensor(symbol_id_valid, dtype=torch.long)

        # 转换为 PyTorch 张量
        dataset = TensorDataset(
            torch.tensor(X_train_reshaped, dtype=torch.float32),
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

            print(f"XLSTM Epoch {epoch + 1}/{self.epochs}, Training Loss: {epoch_loss:.4f}")

            # 验证阶段（如果提供验证数据）
            if X_valid_reshaped is not None and y_valid is not None:
                self.model.eval()
                with torch.no_grad():
                    X_valid_tensor = torch.tensor(X_valid_reshaped, dtype=torch.float32).to(self.device)
                    y_valid_tensor = torch.tensor(y_valid, dtype=torch.float32).to(self.device)
                    symbol_id_valid_tensor = symbol_id_valid_tensor.to(self.device)
                    valid_predictions = self.model(X_valid_tensor)  # (batch_size, num_symbols)
                    valid_predictions_selected = valid_predictions.gather(1,
                                                                          symbol_id_valid_tensor.unsqueeze(1)).squeeze(
                        1)
                    valid_loss = torch.nn.functional.mse_loss(valid_predictions_selected, y_valid_tensor,
                                                              reduction='mean').item()

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

    def predict(self, X_test, symbol_id_test):
        """
        预测函数，接受已经创建好的序列数据和对应的 symbol_id。
        """
        # X_test 已经是 (samples, seq_len, input_size)
        samples = X_test.shape[0]
        X_test_flat = X_test.reshape(-1, self.input_size)

        # 转换为 DataFrame 以便处理
        X_test_df = pd.DataFrame(X_test_flat, columns=FEATURE_NAMES_XLSTM)

        # 填充缺失值为 3
        X_test_imputed = self.imputer.transform(X_test_df)

        # 标准化数据
        X_test_scaled = self.scaler.transform(X_test_imputed)

        # 重塑数据回序列形状
        try:
            X_test_reshaped = X_test_scaled.reshape(samples, self.seq_len, self.input_size)
        except ValueError as e:
            print(f"Error reshaping X_test: {e}")
            print(
                f"Original shape: {X_test_scaled.shape}, Desired shape: ({samples}, {self.seq_len}, {self.input_size})")
            raise

        self.model.eval()
        X_test_tensor = torch.tensor(X_test_reshaped, dtype=torch.float32).to(self.device)
        with torch.no_grad():
            predictions = self.model(X_test_tensor)  # (num_samples, num_symbols)
            symbol_id_tensor = torch.tensor(symbol_id_test, dtype=torch.long).to(self.device)
            predictions_selected = predictions.gather(1, symbol_id_tensor.unsqueeze(1)).cpu().numpy().squeeze(1)
        return predictions_selected


# ----------------- 模型训练函数 -----------------
def train(model_dict, model_name='lgb'):
    for i in range(N_FOLD):
        if TRAINING:
            selected_dates = [date for ii, date in enumerate(train_dates) if ii % N_FOLD != i]
            train_df = df.loc[df['date_id'].isin(selected_dates)]
            valid_df = df.loc[df['date_id'].isin(valid_dates)] if NUM_VALID_DATES > 0 else None

            if model_name == 'xlstm':
                # 创建序列数据并进行前向填充
                X_train, y_train, w_train, symbol_id_train = create_sequences_with_padding(
                    train_df, FEATURE_NAMES_XLSTM, SEQUENCE_LENGTH, symbol_id_to_index
                )
                if NUM_VALID_DATES > 0:
                    X_valid, y_valid, w_valid, symbol_id_valid = create_sequences_with_padding(
                        valid_df, FEATURE_NAMES_XLSTM, SEQUENCE_LENGTH, symbol_id_to_index
                    )
                else:
                    X_valid, y_valid, w_valid, symbol_id_valid = None, None, None, None

                model = model_dict[model_name]
                # XLSTM 模型训练过程
                model.fit(X_train, y_train, symbol_id_train, X_valid, y_valid, symbol_id_valid, sample_weight=w_train)
            else:
                # 对于非序列模型，使用最后一个时间步的特征
                X_train = train_df[FEATURE_NAMES].values
                y_train = train_df['responder_6'].values
                w_train = train_df['weight'].values
                symbol_id_train = train_df['symbol_id'].map(symbol_id_to_index).values

                if NUM_VALID_DATES > 0:
                    X_valid = valid_df[FEATURE_NAMES].values
                    y_valid = valid_df['responder_6'].values
                    w_valid = valid_df['weight'].values
                    symbol_id_valid = valid_df['symbol_id'].map(symbol_id_to_index).values
                else:
                    X_valid, y_valid, w_valid, symbol_id_valid = None, None, None, None

                # 对于其他模型，将 symbol_id 作为特征
                X_train_model = np.hstack((X_train, symbol_id_train.reshape(-1, 1)))
                if X_valid is not None:
                    X_valid_model = np.hstack((X_valid, symbol_id_valid.reshape(-1, 1)))
                else:
                    X_valid_model = None

                model = model_dict[model_name]
                if model_name == 'lgb':
                    model.fit(
                        X_train_model, y_train, sample_weight=w_train,
                        eval_set=[(X_valid_model, y_valid)] if NUM_VALID_DATES > 0 else None,
                        callbacks=[lgb.early_stopping(200), lgb.log_evaluation(10)] if NUM_VALID_DATES > 0 else None
                    )
                elif model_name == 'cbt':
                    if NUM_VALID_DATES > 0:
                        evalset = cbt.Pool(X_valid_model, y_valid, weight=w_valid)
                        model.fit(
                            X_train_model, y_train, sample_weight=w_train,
                            eval_set=[evalset], early_stopping_rounds=200, verbose=10
                        )
                    else:
                        model.fit(X_train_model, y_train, sample_weight=w_train)
                elif model_name == 'xgb':
                    model.fit(
                        X_train_model, y_train, sample_weight=w_train,
                        eval_set=[(X_valid_model, y_valid)] if NUM_VALID_DATES > 0 else None,
                        sample_weight_eval_set=[w_valid] if NUM_VALID_DATES > 0 else None,
                        early_stopping_rounds=200, verbose=10
                    )

            # 保存模型
            joblib.dump(model, os.path.join(MODEL_DIR, f'{model_name}_{i}.model'))
            del X_train, y_train, w_train, X_valid, y_valid, w_valid, symbol_id_train, symbol_id_valid
        else:
            models.append(joblib.load(os.path.join(MODEL_PATH, f'{model_name}_{i}.model')))


# 收集模型的各折预测
def get_fold_predictions(model_names, test_df, feature_names_xlstm, feature_names_other, symbol_id_to_index,
                         sequence_length):
    fold_predictions = {model_name: [] for model_name in model_names}
    # 创建序列数据一次，供所有模型使用，并进行前向填充
    X_test_xlstm, y_test, w_test, symbol_id_test = create_sequences_with_padding(
        test_df, feature_names_xlstm, sequence_length, symbol_id_to_index
    )
    # 对其他模型提取每个序列的最后一个时间步的特征
    X_test_other = X_test_xlstm[:, -1, :]  # shape: (num_sequences, input_size)
    symbol_id_test_other = symbol_id_test  # shape: (num_sequences,)

    for model_name in model_names:
        for i in range(N_FOLD):
            model_path = os.path.join(MODEL_DIR, f'{model_name}_{i}.model')
            model = joblib.load(model_path)
            if model_name == 'xlstm':
                predictions = model.predict(X_test_xlstm, symbol_id_test)
                fold_predictions[model_name].append(predictions)
            else:
                # 将 symbol_id 作为独立特征添加
                X_test_other_model = np.hstack((X_test_other, symbol_id_test_other.reshape(-1, 1)))
                predictions = model.predict(X_test_other_model)
                fold_predictions[model_name].append(predictions)

    return fold_predictions, y_test, w_test


# 计算加权平均的预测
def ensemble_predictions(weights, fold_predictions):
    y_pred = None
    for idx, model_name in enumerate(fold_predictions):
        preds = fold_predictions[model_name]
        avg_pred = np.mean(preds, axis=0)
        avg_pred = avg_pred.squeeze()
        if y_pred is None:
            y_pred = weights[idx] * avg_pred
        else:
            y_pred += weights[idx] * avg_pred
    return y_pred


# 优化权重（使用遗传算法）
def clip_individual(individual, min_val=0.0, max_val=1.0):
    """将个体的基因值限制在[min_val, max_val]范围内。"""
    for i in range(len(individual)):
        if individual[i] < min_val:
            individual[i] = min_val
        elif individual[i] > max_val:
            individual[i] = max_val


# 自定义的交叉操作：交叉后剪裁
def mate_and_clip(ind1, ind2, alpha=0.5):
    offspring1, offspring2 = tools.cxBlend(ind1, ind2, alpha)
    clip_individual(offspring1)
    clip_individual(offspring2)
    return offspring1, offspring2

# 自定义的变异操作：变异后剪裁


def optimize_weights_genetic_algorithm(fold_predictions, y_true, w_true, population_size=50, generations=50):
    """
    使用遗传算法优化模型权重，使加权 R² 分数最大化。
    """
    model_names = list(fold_predictions.keys())
    num_models = len(model_names)

    # 创建 Fitness 类和 Individual 类
    creator.create("FitnessMax", base.Fitness, weights=(1.0,))
    creator.create("Individual", list, fitness=creator.FitnessMax)

    # 初始化工具箱
    toolbox = base.Toolbox()

    # 定义个体：满足权重范围 [0, 1]
    toolbox.register("attr_float", random.uniform, 0.0, 1.0)
    # 定义个体，由多个权重组成
    toolbox.register("individual", tools.initRepeat, creator.Individual, toolbox.attr_float, n=num_models)
    # 定义种群
    toolbox.register("population", tools.initRepeat, list, toolbox.individual)

    # 定义评价函数
    def eval_weights(individual):
        weights = np.array(individual)
        weights /= weights.sum()  # 归一化权重，使其和为 1
        y_pred = ensemble_predictions(weights, fold_predictions)
        score = weighted_r2_score(y_true, y_pred, w_true)
        return (score,)

    # 注册遗传算法的操作
    toolbox.register("evaluate", eval_weights)
    toolbox.register("mate", mate_and_clip, alpha=0.5)
    toolbox.register("mutate", mutate_and_clip, eta=20.0, indpb=1.0 / num_models)
    toolbox.register("select", tools.selTournament, tournsize=3)

    # 创建初始种群
    pop = toolbox.population(n=population_size)

    # 统计信息
    stats = tools.Statistics(lambda ind: ind.fitness.values)
    stats.register("max", np.max)
    stats.register("avg", np.mean)

    # 运行遗传算法
    pop, logbook = algorithms.eaSimple(pop, toolbox, cxpb=0.7, mutpb=0.2, ngen=generations, stats=stats, verbose=False)

    # 获取最佳个体
    top_individuals = tools.selBest(pop, k=1)
    best_weights = np.array(top_individuals[0])
    best_weights /= best_weights.sum()  # 归一化权重
    best_score = top_individuals[0].fitness.values[0]

    # 清除已创建的类，避免重复定义错误
    del creator.FitnessMax
    del creator.Individual

    return best_weights, best_score


# ----------------- 模型字典 -----------------
model_dict = {
    'xlstm': XLSTMWrapper(
        input_size=len(FEATURE_NAMES_XLSTM),
        seq_len=SEQUENCE_LENGTH,
        hidden_size=64,
        num_layers=2,
        num_symbols=num_symbols,
        batch_size=1024,
        lr=0.001,
        epochs=200,
        patience=5,
        device='cuda'
    ),
    'lgb': lgb.LGBMRegressor(n_estimators=5000, device='gpu', gpu_use_dp=True, objective='l2'),
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
num_models = len(model_dict.keys())
def mutate_and_clip(individual, eta=20.0, indpb=1.0 / num_models):
    tools.mutPolynomialBounded(individual, eta=eta, low=0.0, up=1.0, indpb=indpb)
    clip_individual(individual)
    return (individual,)
# 使用测试集评估
if TRAINING:
    test_df = df[df['date_id'].isin(test_dates)]

    model_names = list(model_dict.keys())
    # 修改 get_fold_predictions 函数以返回 y_test 和 w_test
    fold_predictions, y_test, w_test = get_fold_predictions(
        model_names, test_df, FEATURE_NAMES_XLSTM, FEATURE_NAMES_OTHER, symbol_id_to_index, SEQUENCE_LENGTH
    )

    # 检查预测样本数量是否一致
    sample_counts = {model_name: [pred.shape[0] for pred in preds] for model_name, preds in fold_predictions.items()}
    print("各模型预测样本数量：", sample_counts)
    # 确保所有模型的预测样本数量一致
    counts = [preds[0].shape[0] for preds in fold_predictions.values()]
    if not all(count == counts[0] for count in counts):
        raise ValueError("不同模型的预测样本数量不一致。")

    # 优化权重（使用遗传算法）
    optimized_weights, ensemble_r2_score = optimize_weights_genetic_algorithm(fold_predictions, y_test, w_test)

    # 计算各个模型的分数
    model_scores = {}
    for model_name in model_names:
        avg_pred = np.mean(fold_predictions[model_name], axis=0)
        model_scores[model_name] = weighted_r2_score(y_test, avg_pred, w_test)
    print(f"最优模型权重: {dict(zip(model_names, optimized_weights))}")

    y_ensemble_pred = ensemble_predictions(optimized_weights, fold_predictions)
    ensemble_r2_score = weighted_r2_score(y_test, y_ensemble_pred, w_test)
    print(f"Ensemble 的加权 R² 分数: {ensemble_r2_score:.8f}")
