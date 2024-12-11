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

# ----------------- 用户参数 -----------------
ROOT_DIR = r'C:\Users\cyg19\Desktop\kaggle_test'
TRAIN_PATH = os.path.join(ROOT_DIR, 'filtered_train.parquet')
MODEL_DIR = './models_v20_tcts'
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

# ----------------- 定义 TCTS 模型 -----------------
class GRUModel(nn.Module):
    def __init__(self, d_feat=6, hidden_size=64, num_layers=2, dropout=0.0):
        super().__init__()
        self.rnn = nn.GRU(
            input_size=d_feat,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout,
        )
        self.fc_out = nn.Linear(hidden_size, 1)
        self.d_feat = d_feat

    def forward(self, x):
        N = x.shape[0]
        F = self.d_feat
        T = x.shape[1] // F
        x = x.reshape(N,F,T).permute(0,2,1) # [N,T,F]
        out, _ = self.rnn(x)
        return self.fc_out(out[:, -1, :]).squeeze()

class MLPModel(nn.Module):
    def __init__(self, d_feat, hidden_size=256, num_layers=3, dropout=0.0, output_dim=1):
        super().__init__()
        self.mlp = nn.Sequential()
        self.softmax = nn.Softmax(dim=1)
        for i in range(num_layers):
            if i > 0:
                self.mlp.add_module("drop_%d" % i, nn.Dropout(dropout))
            self.mlp.add_module("fc_%d" % i, nn.Linear(d_feat if i == 0 else hidden_size, hidden_size))
            self.mlp.add_module("relu_%d" % i, nn.ReLU())
        self.mlp.add_module("fc_out", nn.Linear(hidden_size, output_dim))
    def forward(self, x):
        out = self.mlp(x).squeeze()
        out = self.softmax(out)
        return out

class TCTSModel:
    def __init__(
        self,
        d_feat=158,
        hidden_size=64,
        num_layers=2,
        dropout=0.0,
        n_epochs=200,
        batch_size=2000,
        early_stop=20,
        loss="mse",
        fore_optimizer="adam",
        weight_optimizer="adam",
        input_dim=360,
        output_dim=5,
        fore_lr=5e-7,
        weight_lr=5e-7,
        steps=3,
        GPU=0,
        target_label=0,
        mode="soft",
        seed=None,
        lowest_valid_performance=0.993,
    ):
        self.d_feat = d_feat
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.dropout = dropout
        self.n_epochs = n_epochs
        self.batch_size = batch_size
        self.early_stop = early_stop
        self.loss = loss
        self.device = torch.device("cuda:%d" % (GPU) if torch.cuda.is_available() and GPU>=0 else "cpu")
        self.use_gpu = (self.device.type == 'cuda')
        self.seed = seed
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.fore_lr = fore_lr
        self.weight_lr = weight_lr
        self.steps = steps
        self.target_label = target_label
        self.mode = mode
        self.lowest_valid_performance = lowest_valid_performance
        self._fore_optimizer = fore_optimizer
        self._weight_optimizer = weight_optimizer
        self.fitted = False

    def loss_fn(self, pred, label, weight, sample_weight):
        # sample_weight shape: [N]
        # weight shape:
        # label shape: [N, output_dim]
        # pred shape: [N]

        if self.mode == "hard":
            loc = torch.argmax(weight, 1)
            loss = (pred - label[np.arange(weight.shape[0]), loc]) ** 2
            loss = loss * sample_weight
            return torch.mean(loss)
        elif self.mode == "soft":
            # (pred - label.transpose(0,1))**2 shape: [output_dim, N]
            # weight.transpose(0,1) same shape
            # sample_weight: [N], need broadcasting to (output_dim,N)
            loss = (pred - label.transpose(0,1))**2
            loss = loss * weight.transpose(0,1)
            loss = loss * sample_weight.unsqueeze(0)
            return torch.mean(loss)
        else:
            raise NotImplementedError("mode {} is not supported!".format(self.mode))

    def train_epoch(self, X_train, Y_train, X_valid, Y_valid, W_train, W_valid):
        # X_train, Y_train, W_train : numpy arrays
        # shuffle training
        indices = np.arange(len(X_train))
        np.random.shuffle(indices)

        task_embedding = torch.zeros([self.batch_size, self.output_dim]).to(self.device)
        task_embedding[:, self.target_label] = 1

        init_fore_model = copy.deepcopy(self.fore_model)
        for p in init_fore_model.parameters():
            p.requires_grad = False
        init_fore_model.to(self.device)

        self.fore_model.train()
        self.weight_model.train()

        for p in self.weight_model.parameters():
            p.requires_grad = False
        for p in self.fore_model.parameters():
            p.requires_grad = True

        # 优化fore_model
        for _ in range(self.steps):
            for i in range(0, len(indices), self.batch_size):
                end_i = min(i+self.batch_size,len(indices))
                if end_i - i < self.batch_size:
                    break
                feature = torch.tensor(X_train[indices[i:end_i]],dtype=torch.float32).to(self.device)
                label = torch.tensor(Y_train[indices[i:end_i]],dtype=torch.float32).to(self.device)
                sample_weight = torch.tensor(W_train[indices[i:end_i]],dtype=torch.float32).to(self.device)

                with torch.no_grad():
                    init_pred = init_fore_model(feature)
                pred = self.fore_model(feature)
                dis = init_pred - label.transpose(0,1)
                weight_feature = torch.cat((feature, dis.transpose(0,1), label, init_pred.view(-1,1), task_embedding),1)
                weight = self.weight_model(weight_feature)
                loss = self.loss_fn(pred, label, weight, sample_weight)

                self.fore_optimizer.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_value_(self.fore_model.parameters(),3.0)
                self.fore_optimizer.step()

        # 优化weight_model
        indices = np.arange(len(X_valid))
        np.random.shuffle(indices)
        for p in self.weight_model.parameters():
            p.requires_grad = True
        for p in self.fore_model.parameters():
            p.requires_grad = False

        for i in range(0,len(indices), self.batch_size):
            end_i = min(i+self.batch_size,len(indices))
            if end_i - i < self.batch_size:
                break
            feature = torch.tensor(X_valid[indices[i:end_i]],dtype=torch.float32).to(self.device)
            label = torch.tensor(Y_valid[indices[i:end_i]],dtype=torch.float32).to(self.device)
            sample_weight = torch.tensor(W_valid[indices[i:end_i]],dtype=torch.float32).to(self.device)

            pred = self.fore_model(feature)
            dis = pred - label.transpose(0,1)
            weight_feature = torch.cat((feature, dis.transpose(0,1), label, pred.view(-1,1), task_embedding),1)
            weight = self.weight_model(weight_feature)
            loc = torch.argmax(weight,1)
            valid_loss = ((pred - label[:,abs(self.target_label)])**2)*sample_weight
            valid_loss = torch.mean(valid_loss)
            loss = torch.mean(valid_loss * torch.log(weight[np.arange(weight.shape[0]),loc]))

            self.weight_optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_value_(self.weight_model.parameters(),3.0)
            self.weight_optimizer.step()

    def test_epoch(self, X_data, Y_data, W_data):
        self.fore_model.eval()
        losses = []
        indices = np.arange(len(X_data))
        for i in range(0,len(indices),self.batch_size):
            end_i = min(i+self.batch_size,len(indices))
            if end_i - i < self.batch_size:
                break
            feature = torch.tensor(X_data[indices[i:end_i]],dtype=torch.float32).to(self.device)
            label = torch.tensor(Y_data[indices[i:end_i]],dtype=torch.float32).to(self.device)
            sample_weight = torch.tensor(W_data[indices[i:end_i]],dtype=torch.float32).to(self.device)
            with torch.no_grad():
                pred = self.fore_model(feature)
                # test loss just mean((pred - label[:,target])**2 * sample_weight)
                loss = (pred - label[:,abs(self.target_label)])**2
                loss = loss * sample_weight
                loss = torch.mean(loss)
            losses.append(loss.item())
        return np.mean(losses)

    def training(self, X_train, Y_train, X_valid, Y_valid, X_test, Y_test, W_train, W_valid, W_test, verbose=True, save_path=None):
        self.fore_model = GRUModel(d_feat=self.d_feat, hidden_size=self.hidden_size, num_layers=self.num_layers, dropout=self.dropout)
        self.weight_model = MLPModel(d_feat=self.input_dim+3*self.output_dim+1, hidden_size=self.hidden_size, num_layers=self.num_layers, dropout=self.dropout, output_dim=self.output_dim)

        if self._fore_optimizer.lower()=="adam":
            self.fore_optimizer = optim.Adam(self.fore_model.parameters(), lr=self.fore_lr)
        else:
            self.fore_optimizer = optim.SGD(self.fore_model.parameters(), lr=self.fore_lr)

        if self._weight_optimizer.lower()=="adam":
            self.weight_optimizer = optim.Adam(self.weight_model.parameters(), lr=self.weight_lr)
        else:
            self.weight_optimizer = optim.SGD(self.weight_model.parameters(), lr=self.weight_lr)

        self.fitted = False
        self.fore_model.to(self.device)
        self.weight_model.to(self.device)

        best_loss = np.inf
        best_epoch = 0
        stop_round = 0

        for epoch in range(self.n_epochs):
            print("Epoch:", epoch)
            print("training...")
            self.train_epoch(X_train, Y_train, X_valid, Y_valid, W_train, W_valid)
            print("evaluating...")
            val_loss = self.test_epoch(X_valid, Y_valid, W_valid)
            test_loss = self.test_epoch(X_test, Y_test, W_test)
            if verbose:
                print("valid %.6f, test %.6f" % (val_loss, test_loss))

            if val_loss < best_loss:
                best_loss = val_loss
                stop_round = 0
                best_epoch = epoch
                torch.save(copy.deepcopy(self.fore_model.state_dict()), save_path + "_fore_model.bin")
                torch.save(copy.deepcopy(self.weight_model.state_dict()), save_path + "_weight_model.bin")
            else:
                stop_round += 1
                if stop_round >= self.early_stop:
                    print("early stop")
                    break

        print("best loss:", best_loss, "@", best_epoch)
        best_param = torch.load(save_path + "_fore_model.bin", map_location=self.device)
        self.fore_model.load_state_dict(best_param)
        best_param = torch.load(save_path + "_weight_model.bin", map_location=self.device)
        self.weight_model.load_state_dict(best_param)
        self.fitted = True
        if self.use_gpu:
            torch.cuda.empty_cache()
        return best_loss

    def fit_tcts(self, X_train, y_train, X_valid=None, y_valid=None, sample_weight_train=None, sample_weight_valid=None):
        # 若无 valid，则以 train 代替 test
        if X_valid is None or y_valid is None:
            X_valid = X_train
            y_valid = y_train
            W_valid = sample_weight_train if sample_weight_valid is None else sample_weight_valid
        else:
            W_valid = np.ones_like(y_valid) if sample_weight_valid is None else sample_weight_valid
        W_train = np.ones_like(y_train) if sample_weight_train is None else sample_weight_train

        # 没有单独的test，这里使用valid当test
        # 实际中建议提供真正的test集
        X_test = X_valid
        Y_test = y_valid
        W_test = W_valid

        if self.seed is not None:
            np.random.seed(self.seed)
            torch.manual_seed(self.seed)

        save_path = os.path.join(MODEL_DIR, 'tcts_model')
        self.training(X_train, y_train, X_valid, y_valid, X_test, Y_test, W_train, W_valid, W_test, verbose=True, save_path=save_path)

    def predict_tcts(self, X_test):
        if not self.fitted:
            raise ValueError("model is not fitted yet!")
        self.fore_model.eval()
        preds = []
        N = X_test.shape[0]
        with torch.no_grad():
            for i in range(0,N,self.batch_size):
                end_i = min(i+self.batch_size,N)
                Xb = torch.tensor(X_test[i:end_i],dtype=torch.float32).to(self.device)
                p = self.fore_model(Xb)
                preds.append(p.cpu().numpy())
        return np.concatenate(preds)

# 封装TCTS模型为与其他模型类似的接口
class TCTSWrapper:
    def __init__(self,
                 input_dim,
                 num_symbols,
                 d_feat=158,
                 hidden_size=64,
                 num_layers=2,
                 dropout=0.0,
                 n_epochs=100,
                 batch_size=1024,
                 early_stop=20,
                 loss="mse",
                 fore_optimizer="adam",
                 weight_optimizer="adam",
                 input_dim_tcts=360,
                 output_dim=5,
                 fore_lr=5e-7,
                 weight_lr=5e-7,
                 steps=3,
                 GPU=0,
                 target_label=0,
                 mode="soft",
                 seed=993,
                 lowest_valid_performance=0.993):
        self.model = TCTSModel(
            d_feat=d_feat,
            hidden_size=hidden_size,
            num_layers=num_layers,
            dropout=dropout,
            n_epochs=n_epochs,
            batch_size=batch_size,
            early_stop=early_stop,
            loss=loss,
            fore_optimizer=fore_optimizer,
            weight_optimizer=weight_optimizer,
            input_dim=input_dim_tcts,
            output_dim=output_dim,
            fore_lr=fore_lr,
            weight_lr=weight_lr,
            steps=steps,
            GPU=GPU,
            target_label=target_label,
            mode=mode,
            seed=seed,
            lowest_valid_performance=lowest_valid_performance,
        )
        self.input_dim = input_dim
        self.num_symbols = num_symbols

    def fit(self, X_train, y_train, symbol_id_train, sample_weight=None, X_valid=None, y_valid=None, symbol_id_valid=None):
        N,T,F = X_train.shape
        X_train_flat = X_train.reshape(N,T*F)
        if X_valid is not None and y_valid is not None:
            Nv,Tv,Fv = X_valid.shape
            X_valid_flat = X_valid.reshape(Nv,Tv*Fv)
        else:
            X_valid_flat = None
            y_valid = None
        self.model.fit_tcts(X_train_flat, y_train, X_valid_flat, y_valid, sample_weight_train=sample_weight, sample_weight_valid=None)

    def predict(self, X_test, symbol_id_test):
        N,T,F = X_test.shape
        X_test_flat = X_test.reshape(N,T*F)
        return self.model.predict_tcts(X_test_flat)

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
            elif model_name == 'tcts':
                # deep learning模型现在也使用sample_weight
                model.fit(
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
            if model_name == 'tcts':
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
    'tcts': TCTSWrapper(
        input_dim=158,
        num_symbols=num_symbols,
        d_feat=158,
        hidden_size=64,
        num_layers=2,
        dropout=0.0,
        n_epochs=100,
        batch_size=1024,
        early_stop=20,
        loss="mse",
        fore_optimizer="adam",
        weight_optimizer="adam",
        input_dim_tcts=360,
        output_dim=5,
        fore_lr=5e-7,
        weight_lr=5e-7,
        steps=3,
        GPU=0,
        target_label=0,
        mode="soft",
        seed=993,
        lowest_valid_performance=0.993
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
