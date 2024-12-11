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

warnings.filterwarnings("ignore")

# 文件路径和参数
ROOT_DIR = r'C:\Users\cyg19\Desktop\kaggle_test'
TRAIN_PATH = os.path.join(ROOT_DIR, 'filtered_train.parquet')
MODEL_DIR = './models_v11_v2'
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
            if str(col_type)[:3]=='int':
                if c_min>np.iinfo(np.int8).min and c_max<np.iinfo(np.int8).max:
                    df[col]=df[col].astype(np.int8)
                elif c_min>np.iinfo(np.int16).min and c_max<np.iinfo(np.int16).max:
                    df[col]=df[col].astype(np.int16)
                elif c_min>np.iinfo(np.int32).min and c_max<np.iinfo(np.int32).max:
                    df[col]=df[col].astype(np.int32)
                elif c_min>np.iinfo(np.int64).min and c_max<np.iinfo(np.int64).max:
                    df[col]=df[col].astype(np.int64)
            else:
                if c_min>np.finfo(np.float16).min and c_max<np.finfo(np.float16).max:
                    if float16_as32:
                        df[col]=df[col].astype(np.float32)
                    else:
                        df[col]=df[col].astype(np.float16)
                elif c_min>np.finfo(np.float32).min and c_max<np.finfo(np.float32).max:
                    df[col]=df[col].astype(np.float32)
                else:
                    df[col]=df[col].astype(np.float64)
    end_mem = df.memory_usage().sum()/1024**2
    print('Memory usage after optimization is: {:.2f} MB'.format(end_mem))
    print('Decreased by {:.1f}%'.format(100*(start_mem-end_mem)/start_mem))
    return df

if TRAINING:
    if os.path.getsize(TRAIN_PATH)>0:
        df=pd.read_parquet(TRAIN_PATH)
        df=reduce_mem_usage(df,False)
        df=df[df['date_id']>=SKIP_DATES].reset_index(drop=True)
        dates=df['date_id'].unique()
        test_dates=dates[-NUM_TEST_DATES:]
        remaining_dates=dates[:-NUM_TEST_DATES]

        valid_dates = remaining_dates[-NUM_VALID_DATES:] if NUM_VALID_DATES>0 else []
        train_dates = remaining_dates[:-NUM_VALID_DATES] if NUM_VALID_DATES>0 else remaining_dates

        symbol_ids = df['symbol_id'].unique()
        num_symbols = len(symbol_ids)
        symbol_id_to_index={symbol_id:idx for idx,symbol_id in enumerate(symbol_ids)}
        index_to_symbol_id={idx:symbol_id for idx,symbol_id in enumerate(symbol_ids)}

        print("已准备好训练、验证和测试数据集。")
    else:
        print(f"训练文件 '{TRAIN_PATH}' 为空。请提供有效的训练数据集。")
        exit()

def weighted_r2_score(y_true,y_pred,weights):
    numerator = np.sum(weights*(y_true - y_pred)**2)
    denominator = np.sum(weights*(y_true - np.average(y_true,weights=weights))**2)
    return 1-(numerator/denominator)

def create_sequences_with_padding(df, feature_names, sequence_length, symbol_id_to_index):
    @numba.njit(parallel=True, fastmath=True)
    def build_sequences_numba(group_features,sequence_length):
        n,feature_num=group_features.shape
        out_len=n-sequence_length+1
        sequences=np.empty((out_len,sequence_length,feature_num),dtype=group_features.dtype)
        for i in numba.prange(out_len):
            for j in range(sequence_length):
                sequences[i,j,:]=group_features[i+j]
        return sequences

    def process_group(symbol_id,group,feature_names,sequence_length,symbol_id_to_index):
        group_features=group[feature_names].values
        group_targets=group['responder_6'].values
        group_weights=group['weight'].values
        n=len(group_features)
        if n==0:
            return [],[],[],[]
        sequences=[]
        targets=[]
        weights=[]
        symbol_ids_seq=[]
        pad_count=sequence_length-1
        feature_num=group_features.shape[1]

        if pad_count>0:
            pad_seq=np.tile(group_features[0].reshape(1,1,feature_num),(pad_count,sequence_length,1))
            pad_target=np.full(pad_count,group_targets[0])
            pad_weight=np.full(pad_count,group_weights[0])
            pad_symbol_id = np.full(pad_count,symbol_id_to_index[symbol_id])
            sequences.append(pad_seq)
            targets.append(pad_target)
            weights.append(pad_weight)
            symbol_ids_seq.append(pad_symbol_id)

        if n>=sequence_length:
            normal_seqs=build_sequences_numba(group_features,sequence_length)
            normal_targets=group_targets[sequence_length-1:]
            normal_weights=group_weights[sequence_length-1:]
            normal_symbol_id=np.full(n-sequence_length+1,symbol_id_to_index[symbol_id])

            sequences.append(normal_seqs)
            targets.append(normal_targets)
            weights.append(normal_weights)
            symbol_ids_seq.append(normal_symbol_id)

        return sequences,targets,weights,symbol_ids_seq

    grouped=df.groupby('symbol_id')
    num_cores=multiprocessing.cpu_count()-1

    results=Parallel(n_jobs=num_cores)(
        delayed(process_group)(symbol_id,group,feature_names,sequence_length,symbol_id_to_index)
        for symbol_id,group in grouped
    )

    all_sequences=[]
    all_targets=[]
    all_weights=[]
    all_symbol_ids_seq=[]

    for sequences,targets,weights,symbol_ids_seq in results:
        if sequences:
            all_sequences.append(np.concatenate(sequences,axis=0))
            all_targets.append(np.concatenate(targets,axis=0))
            all_weights.append(np.concatenate(weights,axis=0))
            all_symbol_ids_seq.append(np.concatenate(symbol_ids_seq,axis=0))

    if all_sequences:
        X=np.concatenate(all_sequences,axis=0)
        y=np.concatenate(all_targets,axis=0)
        w=np.concatenate(all_weights,axis=0)
        symbol_ids_seq=np.concatenate(all_symbol_ids_seq,axis=0)
    else:
        X=np.empty((0,sequence_length,len(feature_names)))
        y=np.empty((0,))
        w=np.empty((0,))
        symbol_ids_seq=np.empty((0,),dtype=int)

    return X,y,w,symbol_ids_seq

# ------------------ XLSTM模型 -------------------
class sLSTMCell(nn.Module):
    #... 略，与原代码相同
    def __init__(self,input_size,hidden_size):
        super(sLSTMCell,self).__init__()
        self.input_size=input_size
        self.hidden_size=hidden_size
        # 定义参数略

    #... 略

class sLSTM(nn.Module):
    #... 略，与原代码相同
    def __init__(self,input_size,seq_len,hidden_size,num_layers,num_symbols):
        super(sLSTM,self).__init__()
        self.layers = nn.ModuleList(
            [sLSTMCell(input_size if i==0 else hidden_size, hidden_size) for i in range(num_layers)]
        )
        self.linear=nn.Linear(hidden_size,num_symbols)
        self.seq_len=seq_len
        self.input_size=input_size
        self.num_symbols=num_symbols

    #... forward 略

class XLSTMWrapper:
    def __init__(self,input_size,seq_len=1,hidden_size=64,num_layers=2,num_symbols=1,dropout=0.0,batch_size=32,
                 lr=0.0001,epochs=100,patience=5,device='cuda'):
        self.device=torch.device(device if torch.cuda.is_available() else 'cpu')
        self.model=sLSTM(input_size=input_size,seq_len=seq_len,hidden_size=hidden_size,num_layers=num_layers,
                         num_symbols=num_symbols).to(self.device)
        self.batch_size=batch_size
        self.lr=lr
        self.epochs=epochs
        self.patience=patience
        self.optimizer=torch.optim.Adam(self.model.parameters(),lr=lr)
        self.loss_fn=torch.nn.MSELoss(reduction='none') # 使用none来应用样本权重
        self.seq_len=seq_len
        self.input_size=input_size

        self.imputer=SimpleImputer(strategy='constant',fill_value=3)
        self.scaler=StandardScaler()

    def fit(self,X_train,y_train,symbol_id_train,X_valid=None,y_valid=None,symbol_id_valid=None,sample_weight=None):
        flat_feature_names=[f"feature_{i:02d}" for i in range(self.input_size)]
        X_train_flat=X_train.reshape(-1,self.input_size)
        X_train_df=pd.DataFrame(X_train_flat,columns=flat_feature_names)
        if X_valid is not None and y_valid is not None:
            X_valid_flat=X_valid.reshape(-1,self.input_size)
            X_valid_df=pd.DataFrame(X_valid_flat,columns=flat_feature_names)
        else:
            X_valid_df=None

        X_train_imputed=self.imputer.fit_transform(X_train_df)
        if X_valid_df is not None:
            X_valid_imputed=self.imputer.transform(X_valid_df)
        else:
            X_valid_imputed=None

        X_train_scaled=self.scaler.fit_transform(X_train_imputed)
        if X_valid_imputed is not None:
            X_valid_scaled=self.scaler.transform(X_valid_imputed)
        else:
            X_valid_scaled=None

        X_train_reshaped=X_train_scaled.reshape(-1,self.seq_len,self.input_size)
        y_train=y_train.reshape(-1)

        if X_valid_scaled is not None:
            X_valid_reshaped=X_valid_scaled.reshape(-1,self.seq_len,self.input_size)
            y_valid=y_valid.reshape(-1)
        else:
            X_valid_reshaped,y_valid=None,None

        dataset=TensorDataset(
            torch.tensor(X_train_reshaped,dtype=torch.float32),
            torch.tensor(y_train,dtype=torch.float32),
            torch.tensor(sample_weight if sample_weight is not None else np.ones(len(y_train)),dtype=torch.float32),
            torch.tensor(symbol_id_train,dtype=torch.long)
        )
        dataloader=DataLoader(dataset,batch_size=self.batch_size,shuffle=True)

        self.model.train()
        best_loss=float('inf')
        no_improve_epochs=0
        best_param=copy.deepcopy(self.model.state_dict())

        for epoch in range(self.epochs):
            epoch_loss=0.0
            for X_batch,y_batch,w_batch,symbol_id_batch in dataloader:
                X_batch,y_batch,w_batch,symbol_id_batch=(
                    X_batch.to(self.device),
                    y_batch.to(self.device),
                    w_batch.to(self.device),
                    symbol_id_batch.to(self.device)
                )
                self.optimizer.zero_grad()
                predictions=self.model(X_batch)
                predictions_selected=predictions.gather(1,symbol_id_batch.unsqueeze(1)).squeeze(1)
                loss=self.loss_fn(predictions_selected,y_batch)
                weighted_loss=(loss*w_batch).mean()
                weighted_loss.backward()
                torch.nn.utils.clip_grad_norm_(self.model.parameters(),max_norm=1.0)
                self.optimizer.step()
                epoch_loss+=weighted_loss.item()

            print(f"XLSTM Epoch {epoch+1}/{self.epochs}, Training Loss: {epoch_loss:.4f}")

            if X_valid_reshaped is not None and y_valid is not None:
                self.model.eval()
                with torch.no_grad():
                    X_valid_tensor=torch.tensor(X_valid_reshaped,dtype=torch.float32).to(self.device)
                    y_valid_tensor=torch.tensor(y_valid,dtype=torch.float32).to(self.device)
                    symbol_id_valid_tensor=torch.tensor(symbol_id_valid,dtype=torch.long).to(self.device)
                    valid_predictions=self.model(X_valid_tensor)
                    valid_predictions_selected=valid_predictions.gather(1,symbol_id_valid_tensor.unsqueeze(1)).squeeze(1)
                    valid_loss=torch.nn.functional.mse_loss(valid_predictions_selected,y_valid_tensor,reduction='mean').item()

                print(f"XLSTM Epoch {epoch+1}/{self.epochs}, Validation Loss: {valid_loss:.4f}")
                if valid_loss<best_loss:
                    best_loss=valid_loss
                    no_improve_epochs=0
                    best_param=copy.deepcopy(self.model.state_dict())
                else:
                    no_improve_epochs+=1
                if no_improve_epochs>=self.patience:
                    print(f"Early stopping at epoch {epoch+1}, best validation loss: {best_loss:.4f}")
                    break
                self.model.train()

        if X_valid_reshaped is not None and y_valid is not None:
            self.model.load_state_dict(best_param)
            torch.save(best_param,os.path.join(MODEL_DIR,'xlstm_best_param.pth'))

        if self.device.type=='cuda':
            torch.cuda.empty_cache()

    def predict(self,X_test,symbol_id_test):
        samples=X_test.shape[0]
        X_test_flat=X_test.reshape(-1,self.input_size)
        X_test_df=pd.DataFrame(X_test_flat,columns=FEATURE_NAMES_XLSTM)
        X_test_imputed=self.imputer.transform(X_test_df)
        X_test_scaled=self.scaler.transform(X_test_imputed)
        X_test_reshaped=X_test_scaled.reshape(samples,self.seq_len,self.input_size)

        self.model.eval()
        X_test_tensor=torch.tensor(X_test_reshaped,dtype=torch.float32).to(self.device)
        with torch.no_grad():
            predictions=self.model(X_test_tensor)
            symbol_id_tensor=torch.tensor(symbol_id_test,dtype=torch.long).to(self.device)
            predictions_selected=predictions.gather(1,symbol_id_tensor.unsqueeze(1)).cpu().numpy().squeeze(1)
        return predictions_selected

def train(model_dict, model_name='lgb'):
    for i in range(N_FOLD):
        if TRAINING:
            selected_dates=[date for ii,date in enumerate(train_dates) if ii%N_FOLD!=i]
            train_df=df.loc[df['date_id'].isin(selected_dates)]
            valid_df=df.loc[df['date_id'].isin(valid_dates)] if NUM_VALID_DATES>0 else None

            if model_name=='xlstm':
                X_train,y_train,w_train,symbol_id_train=create_sequences_with_padding(
                    train_df,FEATURE_NAMES_XLSTM,SEQUENCE_LENGTH,symbol_id_to_index
                )
                if NUM_VALID_DATES>0:
                    X_valid,y_valid,w_valid,symbol_id_valid=create_sequences_with_padding(
                        valid_df,FEATURE_NAMES_XLSTM,SEQUENCE_LENGTH,symbol_id_to_index
                    )
                else:
                    X_valid,y_valid,w_valid,symbol_id_valid=None,None,None,None

                model=model_dict[model_name]
                model.fit(X_train,y_train,symbol_id_train,X_valid,y_valid,symbol_id_valid,sample_weight=w_train)
            else:
                # 其他模型训练逻辑略
                pass

            joblib.dump(model,os.path.join(MODEL_DIR,f'{model_name}_{i}.model'))
            del X_train,y_train,w_train,X_valid,y_valid,w_valid,symbol_id_train,symbol_id_valid

def get_fold_predictions(model_names,test_df,feature_names_xlstm,feature_names_other,symbol_id_to_index,sequence_length):
    fold_predictions={model_name:[] for model_name in model_names}
    X_test_xlstm,y_test,w_test,symbol_id_test=create_sequences_with_padding(
        test_df,feature_names_xlstm,sequence_length,symbol_id_to_index
    )
    X_test_other=X_test_xlstm[:,-1,:]
    symbol_id_test_other=symbol_id_test

    for model_name in model_names:
        for i in range(N_FOLD):
            model_path=os.path.join(MODEL_DIR,f'{model_name}_{i}.model')
            model=joblib.load(model_path)
            if model_name=='xlstm':
                predictions=model.predict(X_test_xlstm,symbol_id_test)
                fold_predictions[model_name].append(predictions)
            else:
                X_test_other_model=np.hstack((X_test_other,symbol_id_test_other.reshape(-1,1)))
                predictions=model.predict(X_test_other_model)
                fold_predictions[model_name].append(predictions)

    return fold_predictions,y_test,w_test

def ensemble_predictions(weights,fold_predictions):
    y_pred=None
    for idx,model_name in enumerate(fold_predictions):
        preds=fold_predictions[model_name]
        avg_pred=np.mean(preds,axis=0)
        avg_pred=avg_pred.squeeze()
        if y_pred is None:
            y_pred=weights[idx]*avg_pred
        else:
            y_pred+=weights[idx]*avg_pred
    return y_pred

def clip_individual(individual,min_val=0.0,max_val=1.0):
    for i in range(len(individual)):
        if individual[i]<min_val:
            individual[i]=min_val
        elif individual[i]>max_val:
            individual[i]=max_val

def mate_and_clip(ind1,ind2,alpha=0.5):
    offspring1,offspring2=tools.cxBlend(ind1,ind2,alpha)
    clip_individual(offspring1)
    clip_individual(offspring2)
    return offspring1,offspring2

def mutate_and_clip(individual,eta=20.0,indpb=1.0):
    tools.mutPolynomialBounded(individual,eta=eta,low=0.0,up=1.0,indpb=indpb)
    clip_individual(individual)
    return (individual,)

def optimize_weights_genetic_algorithm(fold_predictions,y_true,w_true,population_size=50,generations=50):
    model_names=list(fold_predictions.keys())
    num_models=len(model_names)
    creator.create("FitnessMax", base.Fitness, weights=(1.0,))
    creator.create("Individual",list,fitness=creator.FitnessMax)
    toolbox=base.Toolbox()
    toolbox.register("attr_float",random.uniform,0.0,1.0)
    toolbox.register("individual",tools.initRepeat,creator.Individual,toolbox.attr_float,n=num_models)
    toolbox.register("population",tools.initRepeat,list,toolbox.individual)

    def eval_weights(individual):
        weights=np.array(individual)
        weights/=weights.sum()
        y_pred=ensemble_predictions(weights,fold_predictions)
        score=weighted_r2_score(y_true,y_pred,w_true)
        return (score,)

    toolbox.register("evaluate",eval_weights)
    toolbox.register("mate",mate_and_clip,alpha=0.5)
    toolbox.register("mutate",mutate_and_clip,eta=20.0,indpb=1.0/num_models)
    toolbox.register("select",tools.selTournament,tournsize=3)

    pop=toolbox.population(n=population_size)
    stats=tools.Statistics(lambda ind:ind.fitness.values)
    stats.register("max",np.max)
    stats.register("avg",np.mean)

    pop,logbook=algorithms.eaSimple(pop,toolbox,cxpb=0.7,mutpb=0.2,ngen=generations,stats=stats,verbose=False)
    top_individuals=tools.selBest(pop,k=1)
    best_weights=np.array(top_individuals[0])
    best_weights/=best_weights.sum()
    best_score=top_individuals[0].fitness.values[0]

    del creator.FitnessMax
    del creator.Individual

    return best_weights,best_score

model_dict={
    'xlstm':XLSTMWrapper(
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

for model_name in model_dict.keys():
    train(model_dict,model_name)

num_models=len(model_dict.keys())
def mutate_and_clip(individual,eta=20.0,indpb=1.0/num_models):
    tools.mutPolynomialBounded(individual,eta=eta,low=0.0,up=1.0,indpb=indpb)
    clip_individual(individual)
    return (individual,)

if TRAINING:
    test_df=df[df['date_id'].isin(test_dates)]
    model_names=list(model_dict.keys())
    fold_predictions,y_test,w_test=get_fold_predictions(
        model_names,test_df,FEATURE_NAMES_XLSTM,FEATURE_NAMES_OTHER,symbol_id_to_index,SEQUENCE_LENGTH
    )
    sample_counts={model_name:[pred.shape[0] for pred in preds] for model_name,preds in fold_predictions.items()}
    print("各模型预测样本数量：",sample_counts)
    counts=[preds[0].shape[0] for preds in fold_predictions.values()]
    if not all(count==counts[0] for count in counts):
        raise ValueError("不同模型的预测样本数量不一致。")

    optimized_weights,ensemble_r2_score=optimize_weights_genetic_algorithm(fold_predictions,y_test,w_test)

    model_scores={}
    for model_name in model_names:
        avg_pred=np.mean(fold_predictions[model_name],axis=0)
        model_scores[model_name]=weighted_r2_score(y_test,avg_pred,w_test)
    print(f"最优模型权重: {dict(zip(model_names,optimized_weights))}")

    y_ensemble_pred=ensemble_predictions(optimized_weights,fold_predictions)
    ensemble_r2_score=weighted_r2_score(y_test,y_ensemble_pred,w_test)
    print(f"Ensemble 的加权 R² 分数: {ensemble_r2_score:.8f}")
