import os
import joblib
import numpy as np
import pandas as pd
import random
import xgboost as xgb
import catboost as cbt
import lightgbm as lgb
import torch
from torch.utils.data import DataLoader, TensorDataset
from deap import base, creator, tools, algorithms
import copy
from joblib import Parallel, delayed
import numba
import multiprocessing
from tqdm import tqdm
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.autograd import Function

# ----------------- 用户参数 -----------------
ROOT_DIR = r'C:\Users\cyg19\Desktop\kaggle_test'
TRAIN_PATH = os.path.join(ROOT_DIR, 'filtered_train.parquet')
MODEL_DIR = './models_v20'
MODEL_PATH = './pretrained_models'
os.makedirs(ROOT_DIR, exist_ok=True)
os.makedirs(MODEL_DIR, exist_ok=True)
os.makedirs(MODEL_PATH, exist_ok=True)

TRAINING = True
FEATURE_NAMES_XLSTM = [f"feature_{i:02d}" for i in range(79)]
FEATURE_NAMES_OTHER = [f"feature_{i:02d}" for i in range(79)] + ['symbol_id']
NUM_VALID_DATES = 100
NUM_TEST_DATES = 90
SKIP_DATES = 500
N_FOLD = 5
SEQUENCE_LENGTH = 5

def reduce_mem_usage(df, float16_as32=True):
    start_mem = df.memory_usage().sum() / 1024**2
    print('Memory usage of dataframe is {:.2f} MB'.format(start_mem))

    for col in df.columns:
        col_type = df[col].dtype
        if col_type != object and str(col_type)!='category':
            c_min,c_max = df[col].min(),df[col].max()
            if str(col_type)[:3] == 'int':
                if c_min > np.iinfo(np.int8).min and c_max < np.iinfo(np.int8).max:
                    df[col] = df[col].astype(np.int8)
                elif c_min > np.iinfo(np.int16).min and c_max < np.iinfo(np.int16).max:
                    df[col] = df[col].astype(np.int16)
                elif c_min > np.iinfo(np.int32).min and c_max < np.iinfo(np.int32).max:
                    df[col] = df[col].astype(np.int32)
                elif c_min > np.iinfo(np.int64).min and c_max < np.iinfo(np.int64).max:
                    df[col] = df[col].astype(np.int64)
            else:
                if c_min > np.finfo(np.float16).min and c_max < np.finfo(np.float16).max:
                    if float16_as32:
                        df[col] = df[col].astype(np.float32)
                    else:
                        df[col] = df[col].astype(np.float16)
                elif c_min > np.finfo(np.float32).min and c_max < np.finfo(np.float32).max:
                    df[col] = df[col].astype(np.float32)
                else:
                    df[col] = df[col].astype(np.float64)
    end_mem = df.memory_usage().sum() / 1024**2
    print('Memory usage after optimization is: {:.2f} MB'.format(end_mem))
    print('Decreased by {:.1f}%'.format(100 * (start_mem - end_mem) / start_mem))
    return df

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

        symbol_ids = df['symbol_id'].unique()
        symbol_id_to_index = {symbol_id: idx for idx, symbol_id in enumerate(symbol_ids)}
        df['symbol_id'] = df['symbol_id'].map(symbol_id_to_index)
        num_symbols = len(symbol_ids)
        print("已准备好训练、验证和测试数据集。")
    else:
        print(f"训练文件 '{TRAIN_PATH}' 为空。请提供有效的训练数据集。")
        exit()

def weighted_r2_score(y_true, y_pred, weights):
    numerator = np.sum(weights * (y_true - y_pred) ** 2)
    denominator = np.sum(weights * (y_true - np.average(y_true, weights=weights)) ** 2)
    return 1 - (numerator / denominator)

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

        if pad_count > 0:
            pad_seq = np.tile(group_features[0].reshape(1, 1, feature_num), (pad_count, sequence_length, 1))
            pad_target = np.full(pad_count, group_targets[0])
            pad_weight = np.full(pad_count, group_weights[0])
            pad_symbol_id = np.full(pad_count, symbol_id_to_index[symbol_id])
            sequences.append(pad_seq)
            targets.append(pad_target)
            weights.append(pad_weight)
            symbol_ids_seq.append(pad_symbol_id)

        if n >= sequence_length:
            normal_seqs = build_sequences_numba(group_features, sequence_length)
            normal_targets = group_targets[sequence_length - 1:]
            normal_weights = group_weights[sequence_length - 1:]
            normal_symbol_id = np.full(n - sequence_length + 1, symbol_id_to_index[symbol_id])

            sequences.append(normal_seqs)
            targets.append(normal_targets)
            weights.append(normal_weights)
            symbol_ids_seq.append(normal_symbol_id)

        return sequences, targets, weights, symbol_ids_seq

    grouped_items = list(df.groupby('symbol_id'))
    num_cores = multiprocessing.cpu_count() - 1
    results = Parallel(n_jobs=num_cores)(
        delayed(process_group)(symbol_id, group, feature_names, sequence_length, symbol_id_to_index)
        for symbol_id, group in tqdm(grouped_items, desc='Processing sequences')
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

def clip_individual(individual, min_val=0.0, max_val=1.0):
    for i in range(len(individual)):
        if individual[i] < min_val:
            individual[i] = min_val
        elif individual[i] > max_val:
            individual[i] = max_val

def mate_and_clip(ind1, ind2, alpha=0.5):
    offspring1, offspring2 = tools.cxBlend(ind1, ind2, alpha)
    clip_individual(offspring1)
    clip_individual(offspring2)
    return offspring1, offspring2

def mutate_and_clip(individual, eta=20.0, indpb=1.0):
    tools.mutPolynomialBounded(individual, eta=eta, low=0.0, up=1.0, indpb=indpb)
    clip_individual(individual)
    return (individual,)

# ----------------- Tabnet模型中加入sample_weight逻辑 -----------------

class GBN(nn.Module):
    def __init__(self, inp, vbs=1024, momentum=0.01):
        super().__init__()
        self.bn = nn.BatchNorm1d(inp, momentum=momentum)
        self.vbs = vbs
    def forward(self, x):
        if x.size(0) <= self.vbs:
            return self.bn(x)
        else:
            chunk = torch.chunk(x, x.size(0) // self.vbs, 0)
            res = [self.bn(y) for y in chunk]
            return torch.cat(res, 0)

class GLU(nn.Module):
    def __init__(self, inp_dim, out_dim, fc=None, vbs=1024):
        super().__init__()
        if fc:
            self.fc = fc
        else:
            self.fc = nn.Linear(inp_dim, out_dim * 2)
        self.bn = GBN(out_dim * 2, vbs=vbs)
        self.od = out_dim
    def forward(self, x):
        x = self.bn(self.fc(x))
        return torch.mul(x[:, : self.od], torch.sigmoid(x[:, self.od :]))

def make_ix_like(input, dim=0):
    d = input.size(dim)
    rho = torch.arange(1, d + 1, device=input.device, dtype=input.dtype)
    view = [1] * input.dim()
    view[0] = -1
    return rho.view(view).transpose(0, dim)

class SparsemaxFunction(Function):
    @staticmethod
    def forward(ctx, input, dim=-1):
        ctx.dim = dim
        max_val, _ = input.max(dim=dim, keepdim=True)
        input -= max_val
        tau, supp_size = SparsemaxFunction.threshold_and_support(input, dim=dim)
        output = torch.clamp(input - tau, min=0)
        ctx.save_for_backward(supp_size, output)
        return output
    @staticmethod
    def backward(ctx, grad_output):
        supp_size, output = ctx.saved_tensors
        dim = ctx.dim
        grad_input = grad_output.clone()
        grad_input[output == 0] = 0
        v_hat = grad_input.sum(dim=dim) / supp_size.to(output.dtype).squeeze()
        v_hat = v_hat.unsqueeze(dim)
        grad_input = torch.where(output != 0, grad_input - v_hat, grad_input)
        return grad_input, None
    @staticmethod
    def threshold_and_support(input, dim=-1):
        input_srt, _ = torch.sort(input, descending=True, dim=dim)
        input_cumsum = input_srt.cumsum(dim) - 1
        rhos = make_ix_like(input, dim)
        support = rhos * input_srt > input_cumsum
        support_size = support.sum(dim=dim).unsqueeze(dim)
        tau = input_cumsum.gather(dim, support_size - 1)
        tau /= support_size.to(input.dtype)
        return tau, support_size

class FeatureTransformer(nn.Module):
    def __init__(self, inp_dim, out_dim, shared, n_ind, vbs):
        super().__init__()
        self.shared = nn.ModuleList()
        first = True
        if shared:
            self.shared.append(GLU(inp_dim, out_dim, shared[0], vbs=vbs))
            first = False
            for fc in shared[1:]:
                self.shared.append(GLU(out_dim, out_dim, fc, vbs=vbs))
        else:
            self.shared = None
        self.independ = nn.ModuleList()
        if first:
            self.independ.append(GLU(inp_dim, out_dim, vbs=vbs))
        for x in range(first, n_ind):
            self.independ.append(GLU(out_dim, out_dim, vbs=vbs))
        self.scale = float(np.sqrt(0.5))
    def forward(self, x):
        if self.shared:
            x = self.shared[0](x)
            for glu in self.shared[1:]:
                x = torch.add(x, glu(x))
                x = x * self.scale
        for glu in self.independ:
            x = torch.add(x, glu(x))
            x = x * self.scale
        return x

class AttentionTransformer(nn.Module):
    def __init__(self, d_a, inp_dim, relax, vbs=1024):
        super().__init__()
        self.fc = nn.Linear(d_a, inp_dim)
        self.bn = GBN(inp_dim, vbs=vbs)
        self.r = relax
    def forward(self, a, priors):
        a = self.bn(self.fc(a))
        mask = SparsemaxFunction.apply(a * priors)
        priors = priors * (self.r - mask)
        return mask

class DecisionStep(nn.Module):
    def __init__(self, inp_dim, n_d, n_a, shared, n_ind, relax, vbs):
        super().__init__()
        self.atten_tran = AttentionTransformer(n_a, inp_dim, relax, vbs)
        self.fea_tran = FeatureTransformer(inp_dim, n_d + n_a, shared, n_ind, vbs)
    def forward(self, x, a, priors):
        mask = self.atten_tran(a, priors)
        sparse_loss = ((-1)*mask * torch.log(mask + 1e-10)).mean()
        x = self.fea_tran(x * mask)
        return x, sparse_loss

class TabNet(nn.Module):
    def __init__(self, inp_dim=6, out_dim=6, n_d=64, n_a=64, n_shared=2, n_ind=2, n_steps=5, relax=1.2, vbs=1024):
        super().__init__()
        if n_shared > 0:
            self.shared = nn.ModuleList()
            self.shared.append(nn.Linear(inp_dim, 2 * (n_d + n_a)))
            for x in range(n_shared - 1):
                self.shared.append(nn.Linear(n_d + n_a, 2*(n_d+n_a)))
        else:
            self.shared = None
        self.first_step = FeatureTransformer(inp_dim, n_d+n_a, self.shared, n_ind, vbs)
        self.steps = nn.ModuleList()
        for x in range(n_steps-1):
            self.steps.append(DecisionStep(inp_dim, n_d, n_a, self.shared, n_ind, relax, vbs))
        self.fc = nn.Linear(n_d, out_dim)
        self.bn = nn.BatchNorm1d(inp_dim, momentum=0.01)
        self.n_d = n_d
    def forward(self, x, priors):
        x = self.bn(x)
        x_a = self.first_step(x)[:, self.n_d:]
        sparse_loss = []
        out = torch.zeros(x.size(0), self.n_d).to(x.device)
        for step in self.steps:
            x_te, loss = step(x, x_a, priors)
            out += F.relu(x_te[:, :self.n_d])
            x_a = x_te[:, self.n_d:]
            sparse_loss.append(loss)
        return self.fc(out), sum(sparse_loss)

class DecoderStep(nn.Module):
    def __init__(self, inp_dim, out_dim, shared, n_ind, vbs):
        super().__init__()
        self.fea_tran = FeatureTransformer(inp_dim, out_dim, shared, n_ind, vbs)
        self.fc = nn.Linear(out_dim, out_dim)
    def forward(self, x):
        x = self.fea_tran(x)
        return self.fc(x)

class TabNet_Decoder(nn.Module):
    def __init__(self, inp_dim, out_dim, n_shared, n_ind, vbs, n_steps):
        super().__init__()
        self.out_dim = out_dim
        if n_shared > 0:
            self.shared = nn.ModuleList()
            self.shared.append(nn.Linear(inp_dim, 2 * out_dim))
            for x in range(n_shared -1):
                self.shared.append(nn.Linear(out_dim, 2*out_dim))
        else:
            self.shared=None
        self.n_steps=n_steps
        self.steps=nn.ModuleList()
        for x in range(n_steps):
            self.steps.append(DecoderStep(inp_dim, out_dim, self.shared, n_ind, vbs))
    def forward(self, x):
        out = torch.zeros(x.size(0), self.out_dim).to(x.device)
        for step in self.steps:
            out+=step(x)
        return out

class FinetuneModel(nn.Module):
    def __init__(self, input_dim, output_dim, trained_model):
        super().__init__()
        self.model = trained_model
        self.fc = nn.Linear(input_dim, output_dim)
    def forward(self, x, priors):
        return self.fc(self.model(x, priors)[0]).squeeze()

class TabnetModel:
    def __init__(
        self,
        d_feat=158,
        out_dim=64,
        final_out_dim=1,
        batch_size=4096,
        n_d=64,
        n_a=64,
        n_shared=2,
        n_ind=2,
        n_steps=5,
        n_epochs=100,
        pretrain_n_epochs=50,
        relax=1.3,
        vbs=2048,
        seed=993,
        optimizer="adam",
        loss="mse",
        metric="",
        early_stop=20,
        GPU=0,
        pretrain_loss="custom",
        ps=0.3,
        lr=0.01,
        pretrain=True,
        pretrain_file=None
    ):
        print("Initializing TabnetModel...")
        self.d_feat = d_feat
        self.out_dim = out_dim
        self.final_out_dim = final_out_dim
        self.lr = lr
        self.batch_size = batch_size
        self.optimizer = optimizer.lower()
        self.pretrain_loss = pretrain_loss
        self.seed = seed
        self.ps = ps
        self.n_epochs = n_epochs
        self.pretrain_n_epochs = pretrain_n_epochs
        self.device = "cuda:%s" % (GPU) if (torch.cuda.is_available() and GPU>=0) else "cpu"
        self.loss = loss
        self.metric = metric
        self.early_stop = early_stop
        self.pretrain = pretrain
        self.pretrain_file = pretrain_file
        self.fitted = False
        np.random.seed(self.seed)
        torch.manual_seed(self.seed)

        self.tabnet_model = TabNet(inp_dim=self.d_feat, out_dim=self.out_dim, vbs=vbs, relax=relax).to(self.device)
        self.tabnet_decoder = TabNet_Decoder(self.out_dim, self.d_feat, n_shared, n_ind, vbs, n_steps).to(self.device)

        if self.optimizer == "adam":
            self.pretrain_optimizer = optim.Adam(
                list(self.tabnet_model.parameters()) + list(self.tabnet_decoder.parameters()), lr=self.lr
            )
            self.train_optimizer = optim.Adam(self.tabnet_model.parameters(), lr=self.lr)
        elif self.optimizer=="gd":
            self.pretrain_optimizer = optim.SGD(
                list(self.tabnet_model.parameters()) + list(self.tabnet_decoder.parameters()), lr=self.lr)
            self.train_optimizer = optim.SGD(self.tabnet_model.parameters(), lr=self.lr)
        else:
            raise NotImplementedError("optimizer {} is not supported!".format(self.optimizer))

    def mse(self, pred, label):
        loss = (pred - label)**2
        return torch.mean(loss)

    def loss_fn(self, pred, label):
        mask = ~torch.isnan(label)
        if self.loss == "mse":
            return self.mse(pred[mask], label[mask])
        raise ValueError("unknown loss `%s`" % self.loss)

    def metric_fn(self, pred, label):
        mask = torch.isfinite(label)
        if self.metric in ("","loss"):
            return -self.loss_fn(pred[mask], label[mask])
        raise ValueError("unknown metric `%s`" % self.metric)

    def pretrain_loss_fn(self, f_hat, f, S):
        down_mean = torch.mean(f, dim=0)
        down = torch.sqrt(torch.sum(torch.square(f - down_mean), dim=0))
        up = (f_hat - f)*S
        return torch.sum(torch.square(up/down))

    def fit_tabnet(
        self,
        X_train, y_train,
        X_valid=None, y_valid=None,
        sample_weight_train=None,
        sample_weight_valid=None
    ):
        # 加finetune层
        self.tabnet_model = FinetuneModel(self.out_dim, self.final_out_dim, self.tabnet_model).to(self.device)
        best_score = -np.inf
        best_param = None
        best_epoch = 0
        no_improve = 0

        X_train = np.nan_to_num(X_train, nan=0.0)
        y_train = np.nan_to_num(y_train, nan=0.0)
        if X_valid is not None and y_valid is not None:
            X_valid = np.nan_to_num(X_valid, nan=0.0)
            y_valid = np.nan_to_num(y_valid, nan=0.0)

        if sample_weight_train is None:
            sample_weight_train = np.ones_like(y_train)
        if X_valid is not None and y_valid is not None:
            if sample_weight_valid is None:
                sample_weight_valid = np.ones_like(y_valid)

        N = X_train.shape[0]

        for epoch_idx in range(self.n_epochs):
            print(f"Tabnet Epoch {epoch_idx+1}/{self.n_epochs}")
            self.tabnet_model.train()
            indices = np.arange(N)
            np.random.shuffle(indices)
            train_loss = 0.0
            for i in range(0, N, self.batch_size):
                end_i = min(i+self.batch_size, N)
                if end_i - i < self.batch_size:
                    break
                X_batch = torch.tensor(X_train[indices[i:end_i]], dtype=torch.float32).to(self.device)
                y_batch = torch.tensor(y_train[indices[i:end_i]], dtype=torch.float32).to(self.device)
                w_batch = torch.tensor(sample_weight_train[indices[i:end_i]], dtype=torch.float32).to(self.device)

                priors = torch.ones(end_i - i, self.d_feat).to(self.device)
                self.train_optimizer.zero_grad()
                pred = self.tabnet_model(X_batch, priors)
                # 使用sample_weight加权
                loss = ((pred - y_batch)**2 * w_batch).mean()
                loss.backward()
                torch.nn.utils.clip_grad_value_(self.tabnet_model.parameters(),3.0)
                self.train_optimizer.step()
                train_loss += loss.item()

            print(f"Training Loss: {train_loss}")

            # valid
            if X_valid is not None and y_valid is not None:
                self.tabnet_model.eval()
                with torch.no_grad():
                    val_pred = []
                    val_N = X_valid.shape[0]
                    for vi in range(0, val_N, self.batch_size):
                        end_vi = min(vi+self.batch_size, val_N)
                        Xv = torch.tensor(X_valid[vi:end_vi],dtype=torch.float32).to(self.device)
                        Yv = torch.tensor(y_valid[vi:end_vi],dtype=torch.float32).to(self.device)
                        Wv = torch.tensor(sample_weight_valid[vi:end_vi],dtype=torch.float32).to(self.device)
                        priors = torch.ones(end_vi-vi, self.d_feat).to(self.device)
                        pv = self.tabnet_model(Xv, priors)
                        val_pred.append((pv.detach().cpu().numpy(),Yv.cpu().numpy(),Wv.cpu().numpy()))
                    # 计算加权MSE
                    val_pred_concat = np.concatenate([p for p,yv,wv in val_pred])
                    val_y_concat = np.concatenate([yv for p,yv,wv in val_pred])
                    val_w_concat = np.concatenate([wv for p,yv,wv in val_pred])
                    val_mask = ~np.isnan(val_y_concat)
                    val_loss = np.sum((val_pred_concat[val_mask]-val_y_concat[val_mask])**2*val_w_concat[val_mask]) / np.sum(val_w_concat[val_mask])
                    val_score = -val_loss
                print(f"Validation Loss: {val_loss}, Score:{val_score}")
                if val_score > best_score:
                    best_score = val_score
                    best_param = copy.deepcopy(self.tabnet_model.state_dict())
                    no_improve = 0
                    best_epoch = epoch_idx
                else:
                    no_improve += 1
                    if no_improve >= self.early_stop:
                        print("Early Stop.")
                        break

        if best_param is not None:
            self.tabnet_model.load_state_dict(best_param)
        self.fitted = True
        print(f"Best score {best_score} at epoch {best_epoch}")

    def predict_tabnet(self, X_test):
        X_test = np.nan_to_num(X_test, nan=0.0)
        self.tabnet_model.eval()
        preds = []
        N = X_test.shape[0]
        with torch.no_grad():
            for i in range(0,N,self.batch_size):
                end_i = min(i+self.batch_size,N)
                Xb = torch.tensor(X_test[i:end_i],dtype=torch.float32).to(self.device)
                priors = torch.ones(end_i - i, self.d_feat).to(self.device)
                p = self.tabnet_model(Xb, priors)
                preds.append(p.cpu().numpy())
        return np.concatenate(preds)

# 封装 TabnetModel 为与其他模型一致的接口，并纳入 sample_weight
class TabnetWrapper:
    def __init__(self,
                 input_dim,
                 num_symbols,
                 d_feat=158,
                 out_dim=64,
                 final_out_dim=1,
                 batch_size=4096,
                 n_d=64,
                 n_a=64,
                 n_shared=2,
                 n_ind=2,
                 n_steps=5,
                 n_epochs=100,
                 pretrain_n_epochs=50,
                 relax=1.3,
                 vbs=2048,
                 seed=993,
                 optimizer="adam",
                 loss="mse",
                 metric="",
                 early_stop=20,
                 GPU=0,
                 pretrain_loss="custom",
                 ps=0.3,
                 lr=0.01,
                 pretrain=True,
                 pretrain_file=None):
        self.model = TabnetModel(
            d_feat=d_feat,
            out_dim=out_dim,
            final_out_dim=final_out_dim,
            batch_size=batch_size,
            n_d=n_d,
            n_a=n_a,
            n_shared=n_shared,
            n_ind=n_ind,
            n_steps=n_steps,
            n_epochs=n_epochs,
            pretrain_n_epochs=pretrain_n_epochs,
            relax=relax,
            vbs=vbs,
            seed=seed,
            optimizer=optimizer,
            loss=loss,
            metric=metric,
            early_stop=early_stop,
            GPU=GPU,
            pretrain_loss=pretrain_loss,
            ps=ps,
            lr=lr,
            pretrain=pretrain,
            pretrain_file=pretrain_file
        )
        self.input_dim = input_dim
        self.num_symbols = num_symbols

    def fit(self, X_train, y_train, symbol_id_train, sample_weight=None, X_valid=None, y_valid=None, symbol_id_valid=None):
        N,T,F = X_train.shape
        X_train_flat = X_train.reshape(N,T*F)
        if X_valid is not None and y_valid is not None:
            Nv, Tv, Fv = X_valid.shape
            X_valid_flat = X_valid.reshape(Nv,Tv*Fv)
        else:
            X_valid_flat = None
            y_valid = None
        self.model.fit_tabnet(X_train_flat, y_train, X_valid_flat, y_valid,
                              sample_weight_train=sample_weight, sample_weight_valid=None)

    def predict(self, X_test, symbol_id_test):
        N,T,F = X_test.shape
        X_test_flat = X_test.reshape(N,T*F)
        return self.model.predict_tabnet(X_test_flat)

# ----------------- 模型训练、预测、集成流程 -----------------
def train(model_dict, model_name='lgb'):
    for i in range(N_FOLD):
        if TRAINING:
            selected_dates = [date for ii, date in enumerate(train_dates) if ii % N_FOLD != i]
            train_df = df[df['date_id'].isin(selected_dates)]
            X_train, y_train, w_train, symbol_id_train = create_sequences_with_padding(
                train_df, FEATURE_NAMES_XLSTM, SEQUENCE_LENGTH, symbol_id_to_index
            )
            if NUM_VALID_DATES > 0:
                valid_df = df[df['date_id'].isin(valid_dates)]
                X_valid, y_valid, w_valid, symbol_id_valid = create_sequences_with_padding(
                    valid_df, FEATURE_NAMES_XLSTM, SEQUENCE_LENGTH, symbol_id_to_index
                )
            else:
                X_valid, y_valid, w_valid, symbol_id_valid = None, None, None, None

            model = model_dict[model_name]
            if model_name in ['lgb', 'xgb', 'cbt']:
                X_train_last = X_train[:, -1, :]
                X_train_other = np.hstack((X_train_last, symbol_id_train.reshape(-1, 1)))
                if X_valid is not None:
                    X_valid_last = X_valid[:, -1, :]
                    X_valid_other = np.hstack((X_valid_last, symbol_id_valid.reshape(-1, 1)))
                else:
                    X_valid_other = None
                if model_name=='lgb':
                    model.fit(
                        X_train_other, y_train, sample_weight=w_train,
                        eval_set=[(X_valid_other,y_valid)] if NUM_VALID_DATES>0 else None,
                        callbacks=[lgb.early_stopping(200), lgb.log_evaluation(10)] if NUM_VALID_DATES>0 else None
                    )
                elif model_name=='cbt':
                    if NUM_VALID_DATES >0:
                        evalset= cbt.Pool(X_valid_other,y_valid,weight=w_valid)
                        model.fit(
                            X_train_other,y_train,sample_weight=w_train,
                            eval_set=[evalset],early_stopping_rounds=200,verbose=10
                        )
                    else:
                        model.fit(X_train_other,y_train,sample_weight=w_train)
                elif model_name=='xgb':
                    model.fit(
                        X_train_other,y_train,sample_weight=w_train,
                        eval_set=[(X_valid_other,y_valid)] if NUM_VALID_DATES>0 else None,
                        sample_weight_eval_set=[w_valid] if NUM_VALID_DATES>0 else None,
                        early_stopping_rounds=200,verbose=10
                    )
            elif model_name == 'tabnet':
                self_model = model
                self_model.fit(
                    X_train, y_train, symbol_id_train, sample_weight=w_train,
                    X_valid=X_valid, y_valid=y_valid, symbol_id_valid=symbol_id_valid
                )

            joblib.dump(model, os.path.join(MODEL_DIR, f'{model_name}_{i}.model'))
            del X_train, y_train, w_train, symbol_id_train,X_valid,y_valid,w_valid,symbol_id_valid

def get_fold_predictions(model_names, test_df, feature_names_xlstm, feature_names_other, symbol_id_to_index, sequence_length):
    fold_predictions = {model_name: [] for model_name in model_names}
    X_test_xlstm, y_test, w_test, symbol_id_test = create_sequences_with_padding(
        test_df, feature_names_xlstm, sequence_length, symbol_id_to_index
    )
    X_test_last = X_test_xlstm[:,-1,:]
    X_test_other = np.hstack((X_test_last, symbol_id_test.reshape(-1,1)))

    for model_name in model_names:
        for i in range(N_FOLD):
            model_path = os.path.join(MODEL_DIR, f'{model_name}_{i}.model')
            model = joblib.load(model_path)
            if model_name == 'tabnet':
                predictions = model.predict(X_test_xlstm, symbol_id_test)
                fold_predictions[model_name].append(predictions)
            else:
                predictions = model.predict(X_test_other)
                fold_predictions[model_name].append(predictions)
    return fold_predictions, y_test, w_test

def ensemble_predictions(weights, fold_predictions):
    y_pred = None
    for idx, model_name in enumerate(fold_predictions):
        preds = fold_predictions[model_name]
        avg_pred = np.mean(preds, axis=0)
        avg_pred = avg_pred.squeeze()
        if y_pred is None:
            y_pred = weights[idx]*avg_pred
        else:
            y_pred += weights[idx]*avg_pred
    return y_pred

def optimize_weights_genetic_algorithm(fold_predictions, y_true, w_true, population_size=50, generations=50):
    model_names=list(fold_predictions.keys())
    num_models=len(model_names)
    creator.create("FitnessMax", base.Fitness, weights=(1.0,))
    creator.create("Individual", list, fitness=creator.FitnessMax)
    toolbox=base.Toolbox()
    toolbox.register("attr_float",random.uniform,0.0,1.0)
    toolbox.register("individual",tools.initRepeat,creator.Individual,toolbox.attr_float,n=num_models)
    toolbox.register("population",tools.initRepeat,list,toolbox.individual)

    def eval_weights(individual):
        weights=np.array(individual)
        weights/=weights.sum()
        y_pred = ensemble_predictions(weights, fold_predictions)
        score=weighted_r2_score(y_true,y_pred,w_true)
        return (score,)

    toolbox.register("evaluate",eval_weights)
    toolbox.register("mate", mate_and_clip, alpha=0.5)
    toolbox.register("mutate", mutate_and_clip, eta=20.0, indpb=1.0)
    toolbox.register("select", tools.selTournament, tournsize=3)

    pop=toolbox.population(n=population_size)
    stats=tools.Statistics(lambda ind: ind.fitness.values)
    stats.register("max",np.max)
    stats.register("avg",np.mean)

    pop, logbook= algorithms.eaSimple(pop,toolbox,cxpb=0.7,mutpb=0.2,ngen=generations,stats=stats,verbose=False)
    top_individuals=tools.selBest(pop,k=1)
    best_weights=np.array(top_individuals[0])
    best_weights/=best_weights.sum()
    best_score=top_individuals[0].fitness.values[0]

    del creator.FitnessMax
    del creator.Individual

    return best_weights,best_score

# ----------------- 模型字典定义 -----------------
model_dict = {
    'tabnet': TabnetWrapper(
        input_dim=158, # T=2,F=79 => T*F=158
        num_symbols=num_symbols,
        d_feat=158,
        out_dim=64,
        final_out_dim=1,
        batch_size=1024,
        n_d=64,
        n_a=64,
        n_shared=2,
        n_ind=2,
        n_steps=5,
        n_epochs=100,
        pretrain_n_epochs=50,
        relax=1.3,
        vbs=2048,
        seed=993,
        optimizer="adam",
        loss="mse",
        metric="",
        early_stop=20,
        GPU=0,
        pretrain_loss="custom",
        ps=0.3,
        lr=0.01,
        pretrain=True,
        pretrain_file=os.path.join(MODEL_PATH,'tabnet_pretrain.model')
    ),
    'lgb': lgb.LGBMRegressor(n_estimators=5000, device='gpu', gpu_use_dp=True, objective='l2'),
    'xgb': xgb.XGBRegressor(
        n_estimators=5000,
        learning_rate=0.1,
        max_depth=6,
        tree_method='gpu_hist',
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
if TRAINING:
    for model_name in model_dict.keys():
        train(model_dict, model_name)

    test_df = df[df['date_id'].isin(test_dates)]
    model_names = list(model_dict.keys())
    fold_predictions, y_test, w_test = get_fold_predictions(
        model_names,test_df,FEATURE_NAMES_XLSTM,FEATURE_NAMES_OTHER,symbol_id_to_index,SEQUENCE_LENGTH
    )
    sample_counts = {model_name:[pred.shape[0] for pred in preds] for model_name,preds in fold_predictions.items()}
    print("各模型预测样本数量：",sample_counts)
    counts = [preds[0].shape[0] for preds in fold_predictions.values()]
    if not all(count == counts[0] for count in counts):
        raise ValueError("不同模型的预测样本数量不一致。")

    optimized_weights, ensemble_r2_score = optimize_weights_genetic_algorithm(fold_predictions,y_test,w_test)
    model_scores = {}
    for model_name in model_names:
        avg_pred = np.mean(fold_predictions[model_name], axis=0)
        model_scores[model_name] = weighted_r2_score(y_test, avg_pred, w_test)
    print(f"最优模型权重: {dict(zip(model_names,optimized_weights))}")

    y_ensemble_pred = ensemble_predictions(optimized_weights, fold_predictions)
    ensemble_r2_score = weighted_r2_score(y_test,y_ensemble_pred,w_test)
    print(f"Ensemble 的加权 R² 分数: {ensemble_r2_score:.8f}")
