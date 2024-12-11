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

# ----------------- 文件路径和参数 -----------------
ROOT_DIR = r'C:\Users\cyg19\Desktop\kaggle_test'
TRAIN_PATH = os.path.join(ROOT_DIR, 'filtered_train.parquet')
MODEL_DIR = './models_v18'
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

# ----------------- 定义GAT模型，并在fit中使用sample_weight -----------------
class GATModel(nn.Module):
    def __init__(self, d_feat=79, hidden_size=64, num_layers=2, dropout=0.0, base_model="GRU"):
        super().__init__()
        # Base RNN模型: GRU 或 LSTM
        if base_model == "GRU":
            self.rnn = nn.GRU(
                input_size=d_feat,
                hidden_size=hidden_size,
                num_layers=num_layers,
                batch_first=True,
                dropout=dropout,
            )
        elif base_model == "LSTM":
            self.rnn = nn.LSTM(
                input_size=d_feat,
                hidden_size=hidden_size,
                num_layers=num_layers,
                batch_first=True,
                dropout=dropout,
            )
        else:
            raise ValueError("unknown base model name `%s`" % base_model)

        self.hidden_size = hidden_size
        self.d_feat = d_feat
        self.transformation = nn.Linear(self.hidden_size, self.hidden_size)
        self.a = nn.Parameter(torch.randn(self.hidden_size * 2, 1))
        self.a.requires_grad = True
        self.fc = nn.Linear(self.hidden_size, self.hidden_size)
        self.fc_out = nn.Linear(hidden_size, 1)
        self.leaky_relu = nn.LeakyReLU()
        self.softmax = nn.Softmax(dim=1)

    def cal_attention(self, x, y):
        x = self.transformation(x)
        y = self.transformation(y)
        sample_num = x.shape[0]
        dim = x.shape[1]
        e_x = x.expand(sample_num, sample_num, dim)
        e_y = torch.transpose(e_x, 0, 1)
        attention_in = torch.cat((e_x, e_y), 2).view(-1, dim * 2)
        a_t = torch.t(self.a)
        attention_out = a_t.mm(torch.t(attention_in)).view(sample_num, sample_num)
        attention_out = self.leaky_relu(attention_out)
        att_weight = self.softmax(attention_out)
        return att_weight

    def forward(self, x):
        # x: [N, T, F]
        out, _ = self.rnn(x)
        hidden = out[:, -1, :]  # [N, hidden_size]
        att_weight = self.cal_attention(hidden, hidden)  # [N, N]
        hidden = att_weight.mm(hidden) + hidden
        hidden = self.fc(hidden)
        hidden = self.leaky_relu(hidden)
        return self.fc_out(hidden).squeeze()

class GATsWrapper:
    def __init__(self, input_dim, num_symbols, hidden_dim=64, num_layers=2, dropout=0.0, base_model="GRU", batch_size=1024, lr=0.001, epochs=100, patience=5, device='cuda'):
        self.model = GATModel(d_feat=input_dim, hidden_size=hidden_dim, num_layers=num_layers, dropout=dropout, base_model=base_model)
        self.device = torch.device(device if torch.cuda.is_available() else 'cpu')
        self.batch_size = batch_size
        self.lr = lr
        self.epochs = epochs
        self.patience = patience
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=lr)
        self.loss_fn = torch.nn.MSELoss(reduction='none')
        self.num_symbols = num_symbols
        self.model.to(self.device)

    def fit(self, X_train, y_train, symbol_id_train, sample_weight=None, X_valid=None, y_valid=None, symbol_id_valid=None):
        X_train = np.nan_to_num(X_train, nan=3.0)
        y_train = np.nan_to_num(y_train, nan=3.0)
        if sample_weight is None:
            sample_weight = np.ones_like(y_train)
        if X_valid is not None and y_valid is not None:
            X_valid = np.nan_to_num(X_valid, nan=3.0)
            y_valid = np.nan_to_num(y_valid, nan=3.0)

        dataset = TensorDataset(
            torch.tensor(X_train, dtype=torch.float32),
            torch.tensor(y_train, dtype=torch.float32),
            torch.tensor(symbol_id_train, dtype=torch.long),
            torch.tensor(sample_weight, dtype=torch.float32)
        )
        dataloader = DataLoader(dataset, batch_size=self.batch_size, shuffle=True)

        self.model.train()
        best_loss = float('inf')
        no_improve_epochs = 0
        best_param = copy.deepcopy(self.model.state_dict())

        for epoch in tqdm(range(self.epochs), desc='GATs Training Epochs'):
            epoch_loss = 0.0
            self.model.train()
            for X_batch, y_batch, symbol_batch, w_batch in tqdm(dataloader, desc=f"Epoch {epoch + 1}/{self.epochs}", leave=False):
                X_batch = X_batch.to(self.device)
                y_batch = y_batch.to(self.device)
                w_batch = w_batch.to(self.device)
                self.optimizer.zero_grad()

                predictions = self.model(X_batch)
                loss = self.loss_fn(predictions, y_batch)
                weighted_loss = (loss * w_batch).mean()
                weighted_loss.backward()
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
                self.optimizer.step()
                epoch_loss += weighted_loss.item()

            print(f"GATs Epoch {epoch + 1}/{self.epochs}, Training Loss: {epoch_loss:.4f}")

            if X_valid is not None and y_valid is not None:
                self.model.eval()
                with torch.no_grad():
                    X_valid_tensor = torch.tensor(X_valid, dtype=torch.float32).to(self.device)
                    y_valid_tensor = torch.tensor(y_valid, dtype=torch.float32).to(self.device)
                    valid_predictions = self.model(X_valid_tensor)
                    valid_loss = torch.nn.functional.mse_loss(valid_predictions, y_valid_tensor, reduction='mean').item()

                print(f"GATs Epoch {epoch + 1}/{self.epochs}, Validation Loss: {valid_loss:.4f}")
                if valid_loss < best_loss:
                    best_loss = valid_loss
                    no_improve_epochs = 0
                    best_param = copy.deepcopy(self.model.state_dict())
                else:
                    no_improve_epochs += 1

                if no_improve_epochs >= self.patience:
                    print(f"Early stopping at epoch {epoch + 1}, best validation loss: {best_loss:.4f}")
                    break

        if X_valid is not None and y_valid is not None:
            self.model.load_state_dict(best_param)
            torch.save(best_param, os.path.join(MODEL_DIR, 'gats_best_param.pth'))

        if self.device.type == 'cuda':
            torch.cuda.empty_cache()

    def predict(self, X_test, symbol_id_test):
        self.model.eval()
        with torch.no_grad():
            X_test = np.nan_to_num(X_test, nan=3.0)
            X_test_tensor = torch.tensor(X_test, dtype=torch.float32).to(self.device)
            predictions = self.model(X_test_tensor)
            predictions = predictions.cpu().numpy()
        return predictions

# ----------------- 模型训练函数 -----------------
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

                if model_name == 'lgb':
                    model.fit(
                        X_train_other, y_train, sample_weight=w_train,
                        eval_set=[(X_valid_other, y_valid)] if NUM_VALID_DATES > 0 else None,
                        callbacks=[lgb.early_stopping(200), lgb.log_evaluation(10)] if NUM_VALID_DATES > 0 else None
                    )
                elif model_name == 'cbt':
                    if NUM_VALID_DATES > 0:
                        evalset = cbt.Pool(X_valid_other, y_valid, weight=w_valid)
                        model.fit(
                            X_train_other, y_train, sample_weight=w_train,
                            eval_set=[evalset], early_stopping_rounds=200, verbose=10
                        )
                    else:
                        model.fit(X_train_other, y_train, sample_weight=w_train)
                elif model_name == 'xgb':
                    model.fit(
                        X_train_other, y_train, sample_weight=w_train,
                        eval_set=[(X_valid_other, y_valid)] if NUM_VALID_DATES > 0 else None,
                        sample_weight_eval_set=[w_valid] if NUM_VALID_DATES > 0 else None,
                        early_stopping_rounds=200, verbose=10
                    )
            elif model_name == 'gats':
                # 使用 GATs 模型序列直接fit，并考虑sample_weight
                self_model = model
                self_model.fit(
                    X_train, y_train, symbol_id_train, sample_weight=w_train,
                    X_valid=X_valid, y_valid=y_valid, symbol_id_valid=symbol_id_valid
                )

            joblib.dump(model, os.path.join(MODEL_DIR, f'{model_name}_{i}.model'))
            del X_train, y_train, w_train, symbol_id_train, X_valid, y_valid, w_valid, symbol_id_valid

# ----------------- 收集模型的各折预测 -----------------
def get_fold_predictions(model_names, test_df, feature_names_xlstm, feature_names_other, symbol_id_to_index, sequence_length):
    fold_predictions = {model_name: [] for model_name in model_names}
    X_test_xlstm, y_test, w_test, symbol_id_test = create_sequences_with_padding(
        test_df, feature_names_xlstm, sequence_length, symbol_id_to_index
    )
    X_test_last = X_test_xlstm[:, -1, :]
    X_test_other = np.hstack((X_test_last, symbol_id_test.reshape(-1, 1)))

    for model_name in model_names:
        for i in range(N_FOLD):
            model_path = os.path.join(MODEL_DIR, f'{model_name}_{i}.model')
            model = joblib.load(model_path)
            if model_name == 'gats':
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
    'gats': GATsWrapper(
        input_dim=len(FEATURE_NAMES_XLSTM),
        num_symbols=num_symbols,
        hidden_dim=64,
        num_layers=2,
        dropout=0.0,
        base_model="GRU",
        batch_size=1024,
        lr=0.001,
        epochs=100,
        patience=5,
        device='cuda'
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

models = []
for model_name in model_dict.keys():
    train(model_dict, model_name)

num_models = len(model_dict.keys())

if TRAINING:
    test_df = df[df['date_id'].isin(test_dates)]
    model_names = list(model_dict.keys())
    fold_predictions, y_test, w_test = get_fold_predictions(
        model_names,test_df,FEATURE_NAMES_XLSTM,FEATURE_NAMES_OTHER,symbol_id_to_index,SEQUENCE_LENGTH
    )

    sample_counts = {model_name:[pred.shape[0] for pred in preds] for model_name,preds in fold_predictions.items()}
    print("各模型预测样本数量：", sample_counts)
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
    ensemble_r2_score = weighted_r2_score(y_test, y_ensemble_pred, w_test)
    print(f"Ensemble 的加权 R² 分数: {ensemble_r2_score:.8f}")
