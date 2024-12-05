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
from functools import wraps

from deap import base, creator, tools, algorithms

# 忽略某些警告
warnings.filterwarnings("ignore")

# 文件路径和参数
ROOT_DIR = r'C:\Users\cyg19\Desktop\kaggle_test'
TRAIN_PATH = os.path.join(ROOT_DIR, 'filtered_train.parquet')
MODEL_DIR = './models_v13_v2'
MODEL_PATH = './pretrained_models'

os.makedirs(ROOT_DIR, exist_ok=True)
os.makedirs(MODEL_DIR, exist_ok=True)
os.makedirs(MODEL_PATH, exist_ok=True)

# 全局常量
TRAINING = True
FEATURE_NAMES = [f"feature_{i:02d}" for i in range(79)]
FEATURE_NAMES_TCN = [f"feature_{i:02d}" for i in range(79)]
FEATURE_NAMES_OTHER = [f"feature_{i:02d}" for i in range(79)] + ['symbol_id']
NUM_VALID_DATES = 100
NUM_TEST_DATES = 90
SKIP_DATES = 500
N_FOLD = 5
SEQUENCE_LENGTH = 5

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
    """
    创建序列数据，并在序列开始部分进行前向填充。
    """
    sequences = []
    targets = []
    weights = []
    symbol_ids_seq = []

    for symbol_id, group in df.groupby('symbol_id'):
        group = group.sort_values('date_id').reset_index(drop=True)
        group_features = group[feature_names].values
        group_targets = group['responder_6'].values
        group_weights = group['weight'].values

        # 前向填充：用第一个样本填充前 sequence_length -1 个序列
        for i in range(sequence_length - 1):
            seq = np.tile(group_features[0], (sequence_length, 1))  # 复制第一个样本
            target = group_targets[0]  # 使用第一个目标
            weight = group_weights[0]  # 使用第一个权重
            sequences.append(seq)
            targets.append(target)
            weights.append(weight)
            symbol_ids_seq.append(symbol_id_to_index[symbol_id])

        # 正常生成序列
        for i in range(len(group) - sequence_length + 1):
            seq = group_features[i:i + sequence_length]
            target = group_targets[i + sequence_length - 1]
            weight = group_weights[i + sequence_length - 1]
            sequences.append(seq)
            targets.append(target)
            weights.append(weight)
            symbol_ids_seq.append(symbol_id_to_index[symbol_id])

    X = np.array(sequences)
    y = np.array(targets)
    w = np.array(weights)
    symbol_ids_seq = np.array(symbol_ids_seq)
    return X, y, w, symbol_ids_seq

# mypy: allow-untyped-defs
from typing import Any, TypeVar
from torch import _weight_norm, norm_except_dim
from torch.nn.modules import Module
from torch.nn.parameter import Parameter, UninitializedParameter


__all__ = ["WeightNorm", "weight_norm", "remove_weight_norm"]


class WeightNorm:
    name: str
    dim: int

    def __init__(self, name: str, dim: int) -> None:
        if dim is None:
            dim = -1
        self.name = name
        self.dim = dim

    def compute_weight(self, module: Module) -> Any:
        g = getattr(module, self.name + "_g")
        v = getattr(module, self.name + "_v")
        return _weight_norm(v, g, self.dim)

    @staticmethod
    def apply(module, name: str, dim: int) -> "WeightNorm":
        for hook in module._forward_pre_hooks.values():
            if isinstance(hook, WeightNorm) and hook.name == name:
                raise RuntimeError(
                    f"Cannot register two weight_norm hooks on the same parameter {name}"
                )

        if dim is None:
            dim = -1

        fn = WeightNorm(name, dim)

        weight = getattr(module, name)
        if isinstance(weight, UninitializedParameter):
            raise ValueError(
                "The module passed to `WeightNorm` can't have uninitialized parameters. "
                "Make sure to run the dummy forward before applying weight normalization"
            )
        # remove w from parameter list
        del module._parameters[name]

        # add g and v as new parameters and express w as g/||v|| * v
        module.register_parameter(
            name + "_g", Parameter(norm_except_dim(weight, 2, dim).data)
        )
        module.register_parameter(name + "_v", Parameter(weight.data))
        setattr(module, name, fn.compute_weight(module))

        # recompute weight before every forward()
        module.register_forward_pre_hook(fn)

        return fn

    def remove(self, module: Module) -> None:
        weight = self.compute_weight(module)
        delattr(module, self.name)
        del module._parameters[self.name + "_g"]
        del module._parameters[self.name + "_v"]
        setattr(module, self.name, Parameter(weight.data))

    def __call__(self, module: Module, inputs: Any) -> None:
        setattr(module, self.name, self.compute_weight(module))


T_module = TypeVar("T_module", bound=Module)


def weight_norm(module: T_module, name: str = "weight", dim: int = 0) -> T_module:
    WeightNorm.apply(module, name, dim)
    return module


def remove_weight_norm(module: T_module, name: str = "weight") -> T_module:

    for k, hook in module._forward_pre_hooks.items():
        if isinstance(hook, WeightNorm) and hook.name == name:
            hook.remove(module)
            del module._forward_pre_hooks[k]
            return module

    raise ValueError(f"weight_norm of '{name}' not found in {module}")

# ----------------- TCN 模型 -----------------
class Chomp1d(nn.Module):
    def __init__(self, chomp_size):
        super(Chomp1d, self).__init__()
        self.chomp_size = chomp_size

    def forward(self, x):
        return x[:, :, : -self.chomp_size].contiguous()

class TemporalBlock(nn.Module):
    def __init__(self, n_inputs, n_outputs, kernel_size, stride, dilation, padding, dropout=0.2):
        super(TemporalBlock, self).__init__()
        self.conv1 = weight_norm(
            nn.Conv1d(n_inputs, n_outputs, kernel_size, stride=stride, padding=padding, dilation=dilation)
        )
        self.chomp1 = Chomp1d(padding)
        self.relu1 = nn.ReLU()
        self.dropout1 = nn.Dropout(dropout)

        self.conv2 = weight_norm(
            nn.Conv1d(n_outputs, n_outputs, kernel_size, stride=stride, padding=padding, dilation=dilation)
        )
        self.chomp2 = Chomp1d(padding)
        self.relu2 = nn.ReLU()
        self.dropout2 = nn.Dropout(dropout)

        self.net = nn.Sequential(
            self.conv1, self.chomp1, self.relu1, self.dropout1,
            self.conv2, self.chomp2, self.relu2, self.dropout2
        )
        self.downsample = nn.Conv1d(n_inputs, n_outputs, 1) if n_inputs != n_outputs else None
        self.relu = nn.ReLU()
        self.init_weights()

    def init_weights(self):
        self.conv1.weight.data.normal_(0, 0.01)
        self.conv2.weight.data.normal_(0, 0.01)
        if self.downsample is not None:
            self.downsample.weight.data.normal_(0, 0.01)

    def forward(self, x):
        out = self.net(x)
        res = x if self.downsample is None else self.downsample(x)
        return self.relu(out + res)

class TemporalConvNet(nn.Module):
    def __init__(self, num_inputs, num_channels, kernel_size=2, dropout=0.2):
        super(TemporalConvNet, self).__init__()
        layers = []
        num_levels = len(num_channels)
        for i in range(num_levels):
            dilation_size = 2**i
            in_channels = num_inputs if i == 0 else num_channels[i - 1]
            out_channels = num_channels[i]
            layers += [
                TemporalBlock(
                    in_channels,
                    out_channels,
                    kernel_size,
                    stride=1,
                    dilation=dilation_size,
                    padding=(kernel_size - 1) * dilation_size,
                    dropout=dropout,
                )
            ]

        self.network = nn.Sequential(*layers)

    def forward(self, x):
        return self.network(x)

class TCNModel(nn.Module):
    def __init__(self, num_input, output_size, num_channels, kernel_size, dropout):
        super(TCNModel, self).__init__()
        self.tcn = TemporalConvNet(num_input, num_channels, kernel_size, dropout=dropout)
        self.linear = nn.Linear(num_channels[-1], output_size)

    def forward(self, x):
        x = x.transpose(1, 2)  # 转换为 (batch_size, input_size, seq_len)
        output = self.tcn(x)
        output = self.linear(output[:, :, -1])
        return output.squeeze()

class TCNWrapper:
    def __init__(self, input_size, seq_len=1, hidden_size=128, num_layers=2, num_symbols=1, dropout=0.2, batch_size=32,
                 lr=0.001, epochs=100, patience=5, device='cuda'):
        self.device = torch.device(device if torch.cuda.is_available() else 'cpu')
        self.model = TCNModel(
            num_input=input_size,
            output_size=num_symbols,
            num_channels=[hidden_size] * num_layers,
            kernel_size=5,
            dropout=dropout
        ).to(self.device)
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

            print(f"TCN Epoch {epoch + 1}/{self.epochs}, Training Loss: {epoch_loss:.4f}")

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

                print(f"TCN Epoch {epoch + 1}/{self.epochs}, Validation Loss: {valid_loss:.4f}")

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
            torch.save(best_param, os.path.join(MODEL_DIR, 'tcn_best_param.pth'))

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
        X_test_df = pd.DataFrame(X_test_flat, columns=FEATURE_NAMES_TCN)

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

            if model_name == 'tcn':
                # 创建序列数据并进行前向填充
                X_train, y_train, w_train, symbol_id_train = create_sequences_with_padding(
                    train_df, FEATURE_NAMES_TCN, SEQUENCE_LENGTH, symbol_id_to_index
                )
                if NUM_VALID_DATES > 0:
                    X_valid, y_valid, w_valid, symbol_id_valid = create_sequences_with_padding(
                        valid_df, FEATURE_NAMES_TCN, SEQUENCE_LENGTH, symbol_id_to_index
                    )
                else:
                    X_valid, y_valid, w_valid, symbol_id_valid = None, None, None, None

                model = model_dict[model_name]
                # TCN 模型训练过程
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
def get_fold_predictions(model_names, test_df, feature_names_tcn, feature_names_other, symbol_id_to_index,
                         sequence_length):
    fold_predictions = {model_name: [] for model_name in model_names}
    # 创建序列数据一次，供所有模型使用，并进行前向填充
    X_test_tcn, y_test, w_test, symbol_id_test = create_sequences_with_padding(
        test_df, feature_names_tcn, sequence_length, symbol_id_to_index
    )
    # 对其他模型提取每个序列的最后一个时间步的特征
    X_test_other = X_test_tcn[:, -1, :]  # shape: (num_sequences, input_size)
    symbol_id_test_other = symbol_id_test  # shape: (num_sequences,)

    for model_name in model_names:
        for i in range(N_FOLD):
            model_path = os.path.join(MODEL_DIR, f'{model_name}_{i}.model')
            model = joblib.load(model_path)
            if model_name == 'tcn':
                predictions = model.predict(X_test_tcn, symbol_id_test)
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
    'tcn': TCNWrapper(
        input_size=len(FEATURE_NAMES_TCN),
        seq_len=SEQUENCE_LENGTH,
        hidden_size=128,
        num_layers=2,
        num_symbols=num_symbols,
        dropout=0.2,
        batch_size=32,
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
        model_names, test_df, FEATURE_NAMES_TCN, FEATURE_NAMES_OTHER, symbol_id_to_index, SEQUENCE_LENGTH
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
