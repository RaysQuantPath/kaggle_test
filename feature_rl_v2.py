#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import sys
import math
import numpy as np
import pandas as pd
import random
from xgboost import XGBRegressor
from numba import njit, prange  # 导入Numba

##############################################################################
# ------------------- 全局配置 ----------------------------------------------
##############################################################################
ROOT_DIR = r'/well/ludwig/users/mil024/YY/'
TRAIN_PATH = os.path.join(ROOT_DIR, 'train.parquet')
os.makedirs(ROOT_DIR, exist_ok=True)

TRAINING = True
NUM_VALID_DATES = 340
NUM_TEST_DATES = 169
SKIP_DATES = 510
N_FOLD = 4
ORTH_LAMBDA = 0.05

FEATURE_CSV_PATH = '/well/ludwig/users/mil024/YY/features.csv'  # 如果要用标签相似度
USE_MULTI_GPU_DEVICES = '0,1,2,3'  # 可改为 '0,1,2,3,4' 等

##############################################################################
# ------------------- 内存优化 -----------------------------------------------
##############################################################################
def reduce_mem_usage(df, float16_as32=True):
    start_mem = df.memory_usage().sum() / 1024**2
    print(f"Memory usage of dataframe is {start_mem:.2f} MB")
    for col in df.columns:
        col_type = df[col].dtype
        if col_type != object and str(col_type) != 'category':
            c_min = df[col].min()
            c_max = df[col].max()
            if str(col_type)[:3] == 'int':
                if c_min > np.iinfo(np.int8).min and c_max < np.iinfo(np.int8).max:
                    df[col] = df[col].astype(np.int8)
                elif c_min > np.iinfo(np.int16).min and c_max < np.iinfo(np.int16).max:
                    df[col] = df[col].astype(np.int16)
                elif c_min > np.iinfo(np.int32).min and c_max < np.iinfo(np.int32).max:
                    df[col] = df[col].astype(np.int32)
                else:
                    df[col] = df[col].astype(np.int64)
            else:
                if float16_as32:
                    if c_min > np.finfo(np.float16).min and c_max < np.finfo(np.float16).max:
                        df[col] = df[col].astype(np.float32)
                    elif c_min > np.finfo(np.float32).min and c_max < np.finfo(np.float32).max:
                        df[col] = df[col].astype(np.float32)
                    else:
                        df[col] = df[col].astype(np.float64)
                else:
                    if c_min > np.finfo(np.float16).min and c_max < np.finfo(np.float16).max:
                        df[col] = df[col].astype(np.float16)
                    elif c_min > np.finfo(np.float32).min and c_max < np.finfo(np.float32).max:
                        df[col] = df[col].astype(np.float32)
                    else:
                        df[col] = df[col].astype(np.float64)
    end_mem = df.memory_usage().sum()/1024**2
    print(f"Memory usage after optimization is {end_mem:.2f} MB")
    print(f"Decreased by {(100*(start_mem-end_mem)/start_mem):.1f}%")
    return df

##############################################################################
# ------------------- (可选) 标签相似度 --------------------------------------
##############################################################################

@njit(parallel=True)
def compute_tag_similarity_matrix_numba(feats, arr):
    num_feats = len(feats)
    sim_matrix = {}
    for i in prange(num_feats):
        f1 = feats[i]
        sim_matrix[f1] = {}
        for j in range(num_feats):
            f2 = feats[j]
            overlap = 0
            for k in range(arr.shape[1]):
                overlap += arr[i][k] & arr[j][k]
            sim_matrix[f1][f2] = float(overlap)
    return sim_matrix

def compute_tag_similarity_matrix(tag_df):
    feats = tag_df.index.tolist()
    arr = tag_df.astype(int).values
    sim_matrix = compute_tag_similarity_matrix_numba(feats, arr)
    return sim_matrix

##############################################################################
# ---------------- OperatorTree & evaluate_operator_tree ---------------------
##############################################################################
class OperatorTree:
    def __init__(self, operator=None, children=None, depth=1):
        self.operator = operator
        self.children = children if children else []
        self.depth = depth

    def __repr__(self):
        return f"OpTree(op={self.operator}, depth={self.depth}, kids={self.children})"

def evaluate_operator_tree(tree, df):
    op = tree.operator
    children = tree.children
    if op is None and len(children) == 1 and isinstance(children[0], str):
        return df[children[0]].values
    child_vals = []
    for c in children:
        if isinstance(c, OperatorTree):
            child_vals.append(evaluate_operator_tree(c, df))
        elif isinstance(c, str):
            child_vals.append(df[c].values)
        else:
            raise ValueError("Unknown child in operator_tree.")
    if op in ['+','-','*','/']:
        if len(child_vals) != 2:
            raise ValueError(f"Binary {op} => need 2 children.")
        c1, c2 = child_vals
        if op == '+':
            return c1 + c2
        elif op == '-':
            return c1 - c2
        elif op == '*':
            return c1 * c2
        elif op == '/':
            return np.where(np.abs(c2) < 1e-9, 0, c1/(c2+1e-9))
    else:
        # unary
        if len(child_vals) != 1:
            raise ValueError(f"Unary {op} => 1 child.")
        c = child_vals[0]
        if op == 'log':
            return np.log(np.abs(c)+1e-9)
        elif op == 'exp':
            return np.exp(c)
        elif op == 'sqrt':
            return np.sqrt(np.abs(c))
        elif op == 'square':
            return c**2
        else:
            raise NotImplementedError(f"Unsupported operator {op}")

##############################################################################
# ---------------- Weighted R² + CV + (XGBoost) ------------------------------
##############################################################################
def weighted_r2_score(y_true, y_pred, w):
    numerator = np.sum(w*(y_true-y_pred)**2)
    denominator = np.sum(w*(y_true**2)) + 1e-15
    return 1 - numerator/denominator

def evaluate_with_cv(X, y, w, model_fn, n_splits=4, random_state=42):
    from sklearn.model_selection import KFold
    kf = KFold(n_splits=n_splits, shuffle=True, random_state=random_state)
    scores = []
    for train_idx, val_idx in kf.split(X):
        X_train, X_val = X[train_idx], X[val_idx]
        y_train, y_val = y[train_idx], y[val_idx]
        w_train, w_val = w[train_idx], w[val_idx]
        model = model_fn()
        # ----------------- 保持原有 fit() 调用结构，不额外改动 -----------------
        model.fit(X_train, y_train,
                  sample_weight=w_train,
                  eval_set=[(X_val, y_val)],
                  early_stopping_rounds=100,
                  verbose=False)
        y_pred = model.predict(X_val)
        sc = weighted_r2_score(y_val, y_pred, w_val)
        scores.append(sc)
    return np.mean(scores)

# ----------------- 改动 2：这里改用 XGBoost 替换 CatBoost -----------------
def default_model_fn():
    return XGBRegressor(
        n_estimators=500,
        max_depth=5,
        learning_rate=0.001,
        tree_method='hist',        # 使用直方图加速
        objective='reg:squarederror',
        device = "cuda",                # 多GPU
        # ---------------- 可添加更多超参数 ----------------
        subsample=0.8,
        colsample_bytree=0.8,
        reg_alpha=0.09,
        reg_lambda=0.07,
        gamma=4,
        n_jobs=-1,
        min_child_weight=7
    )

@njit(parallel=True)
def compute_orthogonality_penalty_numba(new_col, X_existing, orth_lambda, num_features):
    if num_features == 0:
        return 0.0
    new_centered = new_col - np.mean(new_col)
    denom_new = np.std(new_centered) + 1e-15
    max_corr_abs = 0.0
    for j in prange(num_features):
        col_j = X_existing[:, j]
        col_centered = col_j - np.mean(col_j)
        denom_j = np.std(col_centered) + 1e-15
        corr = (new_centered * col_centered).mean() / (denom_new * denom_j)
        if abs(corr) > max_corr_abs:
            max_corr_abs = abs(corr)
    penalty = orth_lambda * max_corr_abs
    return penalty

def compute_orthogonality_penalty(new_col, X_existing, orth_lambda=0.05):
    num_features = X_existing.shape[1]
    penalty = compute_orthogonality_penalty_numba(new_col, X_existing, orth_lambda, num_features)
    return penalty

##############################################################################
# ------- 在二元运算时若 use_similarity=True, 用相似度加权“右侧” feature选择 ----
##############################################################################
@njit(parallel=True)
def compute_pair_similarity_numba(leaves, right_feat, sim_matrix):
    if len(leaves) == 0:
        return 1.0
    total = 0.0
    count = 0
    for lf in prange(len(leaves)):
        leaf = leaves[lf]
        if leaf in sim_matrix and right_feat in sim_matrix[leaf]:
            total += sim_matrix[leaf][right_feat]
            count += 1
    if count == 0:
        return 1.0
    return (total / count + 1.0)

def compute_pair_similarity(left_tree, right_feat, sim_matrix):
    leaves = set()
    def dfs(t):
        if t.operator is None and len(t.children) == 1 and isinstance(t.children[0], str):
            leaves.add(t.children[0])
        else:
            for ch in t.children:
                if isinstance(ch, OperatorTree):
                    dfs(ch)
                elif isinstance(ch, str):
                    leaves.add(ch)
    dfs(left_tree)
    leaves = list(leaves)  # 转换为列表以便与Numba兼容
    similarity = compute_pair_similarity_numba(leaves, right_feat, sim_matrix)
    return similarity

##############################################################################
# --------- ActionSpace: 统一管理 (unary_ops, binary_ops, features, finish) --
##############################################################################
class ActionSpace:
    def __init__(self, unary_ops, binary_ops, feature_list, finish_id=999999):
        self.unary_ops = unary_ops
        self.binary_ops = binary_ops
        self.feature_list = feature_list
        self.finish_id = finish_id
        self.mapping = {}
        current_id = 0
        # unary
        for i, op in enumerate(self.unary_ops):
            self.mapping[current_id] = ('unary', i)
            current_id += 1
        # binary
        for i, op in enumerate(self.binary_ops):
            self.mapping[current_id] = ('binary', i)
            current_id += 1
        # features
        for i, f in enumerate(self.feature_list):
            self.mapping[current_id] = ('feature', i)
            current_id += 1
        # finish
        self.mapping[current_id] = ('finish', None)
        self.finish_id = current_id

    def size(self):
        return len(self.mapping)

    def decode(self, action_id):
        return self.mapping[action_id]

    def is_finish(self, action_id):
        return (action_id == self.finish_id)

    def all_actions(self):
        return list(self.mapping.keys())

##############################################################################
# ---------------- ExpressionEnv ---------------------------------------------
##############################################################################
class ExpressionEnv:
    """
    在 step(action)：
     - 若 'finish' => done
     - 若 'feature' => create leaf
     - 若 'unary' => wrap current_tree
     - 若 'binary' => merge current_tree with 右子树(特征可按相似度加权采样).
     - reward = WeightedR²增量 - 正交度
     - depth>=max_depth => done
    """
    def __init__(
        self,
        df,
        feature_list,
        target_col='responder_6',
        unary_ops=['log','exp','sqrt','square'],
        binary_ops=['+','-','*','/'],
        max_depth=3,
        orth_lambda=0.05,
        model_fn= default_model_fn,
        finish_id=999999,
        n_splits=4,
        random_state=42,
        sim_matrix=None,
        use_similarity=False
    ):
        self.df = df
        self.target_col = target_col
        self.y = df[target_col].values
        self.w = np.ones(len(df), dtype=np.float32)
        self.unary_ops = unary_ops
        self.binary_ops = binary_ops
        self.feature_list = feature_list
        self.max_depth = max_depth
        self.orth_lambda = orth_lambda
        self.model_fn = model_fn
        self.finish_id = finish_id
        self.n_splits = n_splits
        self.random_state = random_state
        self.sim_matrix = sim_matrix
        self.use_similarity = use_similarity

        self.action_space = ActionSpace(unary_ops, binary_ops, feature_list, finish_id)
        self.current_tree = None
        self.depth = 0
        self.X_current = np.zeros((len(df),0), dtype=np.float32)
        self.current_score = 0.0
        self.prev_score = 0.0

    def clone(self):
        import copy
        new_env = ExpressionEnv(
            df = self.df,
            feature_list = self.feature_list,
            target_col = self.target_col,
            unary_ops = self.unary_ops,
            binary_ops = self.binary_ops,
            max_depth = self.max_depth,
            orth_lambda = self.orth_lambda,
            model_fn = self.model_fn,
            finish_id = self.finish_id,
            n_splits = self.n_splits,
            random_state = self.random_state,
            sim_matrix = self.sim_matrix,
            use_similarity = self.use_similarity
        )
        new_env.current_tree = copy.deepcopy(self.current_tree)
        new_env.depth = self.depth
        new_env.X_current = np.copy(self.X_current)
        new_env.current_score = self.current_score
        new_env.prev_score = self.prev_score
        return new_env

    def step(self, action_id):
        done = False
        reward = 0.0
        cat, idx = self.action_space.decode(action_id)
        if cat == 'finish':
            done = True
            reward = self.current_score - self.prev_score
            self.prev_score = self.current_score
            return self.observation(), reward, done
        if self.depth >= self.max_depth:
            done = True
            reward = self.current_score - self.prev_score
            return self.observation(), reward, done

        from copy import deepcopy
        new_tree = None
        if cat == 'feature':
            feat_name = self.feature_list[idx]
            leaf = OperatorTree(None, [feat_name], depth=self.depth+1)
            if self.current_tree is None:
                new_tree = leaf
            else:
                # 默认用 '+'
                new_tree = OperatorTree('+', [self.current_tree, leaf],
                                        depth=max(self.current_tree.depth, leaf.depth)+1)
        elif cat == 'unary':
            op_name = self.unary_ops[idx]
            if self.current_tree is None:
                leaf = OperatorTree(None, [self.feature_list[0]], depth=1)
                new_tree = OperatorTree(op_name, [leaf], depth=2)
            else:
                new_tree = OperatorTree(op_name, [self.current_tree],
                                        depth=self.current_tree.depth+1)
        elif cat == 'binary':
            op_name = self.binary_ops[idx]
            # 右侧特征 => 如果 use_similarity 且已有 current_tree => 加权随机
            if (self.use_similarity) and (self.current_tree is not None) and (self.sim_matrix is not None):
                # 根据 compute_pair_similarity
                sim_list = []
                for ft in self.feature_list:
                    s = compute_pair_similarity(self.current_tree, ft, self.sim_matrix)
                    sim_list.append(s)
                sim_arr = np.array(sim_list, dtype=float)
                sim_sum = sim_arr.sum()
                if sim_sum < 1e-9:
                    # fallback => uniform
                    feat_name = random.choice(self.feature_list)
                else:
                    # 轮盘选择
                    rnum = random.random() * sim_sum
                    cumsum = 0.0
                    for i, ft in enumerate(self.feature_list):
                        cumsum += sim_arr[i]
                        if cumsum >= rnum:
                            feat_name = ft
                            break
            else:
                # 否则 => uniform pick
                feat_name = random.choice(self.feature_list)

            leaf = OperatorTree(None, [feat_name], depth=self.depth+1)
            if self.current_tree is None:
                new_tree = leaf
            else:
                new_tree = OperatorTree(op_name, [self.current_tree, leaf],
                                        depth=max(self.current_tree.depth, leaf.depth)+1)
        else:
            raise ValueError(f"Unknown cat={cat}")

        # Evaluate => WeightedR² - orth
        new_col = evaluate_operator_tree(new_tree, self.df)
        penalty_orth = compute_orthogonality_penalty(new_col, self.X_current, self.orth_lambda)
        X_new = np.column_stack([self.X_current, new_col])
        new_score = self._cv_score(X_new)
        delta = new_score - self.current_score
        reward = delta - penalty_orth
        self.depth = new_tree.depth
        self.current_tree = new_tree
        self.X_current = X_new
        self.prev_score = self.current_score
        self.current_score = new_score
        if self.depth >= self.max_depth:
            done = True
        return self.observation(), reward, done

    def observation(self):
        return {"num_cols": self.X_current.shape[1],
                "depth": self.depth,
                "score": self.current_score}

    def legal_actions(self):
        if self.depth >= self.max_depth:
            return [self.finish_id]
        return list(range(self.action_space.size()))

    def terminal(self):
        return (self.depth >= self.max_depth)

    def _cv_score(self, X_new):
        if X_new.shape[1] == 0:
            return 0.0
        from sklearn.model_selection import KFold
        sc = evaluate_with_cv(X_new, self.y, self.w, self.model_fn,
                              n_splits=self.n_splits, random_state=self.random_state)
        return sc

##############################################################################
# ----------------- MCTS Node & pipeline -------------------------------------
##############################################################################
class MCTSNode:
    def __init__(self, prior=1.0):
        self.visit_count = 0
        self.value_sum = 0
        self.prior = prior
        self.reward = 0
        self.children = {}

    def expanded(self):
        return len(self.children) > 0

    def value(self):
        if self.visit_count == 0:
            return 0
        return self.value_sum / self.visit_count

def ucb_score(parent, child, c_puct=1.4):
    if child.visit_count == 0:
        return math.inf
    return child.value() + c_puct * child.prior * math.sqrt(parent.visit_count) / (1 + child.visit_count)

def select_child(node):
    best_a = None
    best_c = None
    best_s = -1e9
    for a, cnode in node.children.items():
        s = ucb_score(node, cnode)
        if s > best_s:
            best_s = s
            best_a = a
            best_c = cnode
    return best_a, best_c

def expand_node_mcts(node, env):
    acts = env.legal_actions()
    if len(acts) == 0:
        return
    p = 1 / len(acts)
    for a in acts:
        node.children[a] = MCTSNode(prior=p)

def backpropagate(path, val, discount=1.0):
    for nd in reversed(path):
        nd.visit_count += 1
        nd.value_sum += val
        val = nd.reward + discount*val

def rollout(sim_env, rollout_steps=1):
    total = 0.0
    done = False
    for _ in range(rollout_steps):
        if done:
            break
        acts = sim_env.legal_actions()
        if len(acts) == 0:
            break
        a = random.choice(acts)
        obs, rew, done = sim_env.step(a)
        total += rew
        if done:
            break
    return total

def run_mcts(env, root, n_sim=50, c_puct=1.4, rollout_steps=1):
    for _ in range(n_sim):
        clone_env = env.clone()
        node = root
        path = [node]
        while node.expanded():
            a, cnode = select_child(node)
            obs, rew, done = clone_env.step(a)
            cnode.reward = rew
            path.append(cnode)
            node = cnode
            if done:
                break
        if (not clone_env.terminal()) and (not node.expanded()):
            expand_node_mcts(node, clone_env)
            val = rollout(clone_env, rollout_steps=rollout_steps)
        else:
            val = 0
        backpropagate(path, val, discount=1.0)

def best_child(node):
    best_vis = -1
    best_a = None
    for a, c in node.children.items():
        if c.visit_count > best_vis:
            best_vis = c.visit_count
            best_a = a
    return best_a

def mcts_search_expression(env, n_sim=50, max_steps=50):
    root = MCTSNode(prior=1.0)
    expand_node_mcts(root, env)
    total_reward = 0.0
    steps = 0
    while True:
        run_mcts(env.clone(), root, n_sim=n_sim, rollout_steps=1)
        a = best_child(root)
        obs, rew, done = env.step(a)
        total_reward += rew
        steps += 1
        if a in root.children:
            root = root.children[a]
        else:
            break
        if done or steps >= max_steps:
            break
    return total_reward, steps

def train_mcts_feature_engineer(
    df,
    target_col='responder_6',
    unary_ops=['log','exp','sqrt','square'],
    binary_ops=['+','-','*','/'],
    feature_list=None,
    max_depth=3,
    num_episodes=5,
    n_sim=50,
    max_steps_per_episode=50,
    sim_matrix=None,
    use_similarity=False
):
    if feature_list is None:
        feature_list = [c for c in df.columns if c.startswith('feature_')]
    final_exprs = []
    for ep in range(num_episodes):
        env = ExpressionEnv(
            df = df,
            feature_list = feature_list,
            target_col = target_col,
            unary_ops = unary_ops,
            binary_ops = binary_ops,
            max_depth = max_depth,
            finish_id = 999999,
            n_splits = 4,
            random_state = 42,
            sim_matrix = sim_matrix,
            use_similarity = use_similarity
        )
        ep_reward, ep_steps = mcts_search_expression(env, n_sim=n_sim, max_steps=max_steps_per_episode)
        print(f"Episode {ep+1}/{num_episodes}, reward={ep_reward:.4f}, steps={ep_steps}, final_depth={env.depth}")
        final_exprs.append(env.current_tree)
    return final_exprs

##############################################################################
# ------------------------------ 主入口 --------------------------------------
##############################################################################
if TRAINING:
    # 1) 相似度矩阵: 读 features.csv => sim_matrix
    if os.path.exists(FEATURE_CSV_PATH):
        tag_csv = pd.read_csv(FEATURE_CSV_PATH)
        tag_csv.set_index('feature', inplace=True)
        for c in tag_csv.columns:
            tag_csv[c] = tag_csv[c].astype(bool)
        sim_matrix = compute_tag_similarity_matrix(tag_csv)
        use_similarity = True
        print("相似度矩阵已读取并计算.")
    else:
        sim_matrix = None
        use_similarity = False
        print(f"未找到 {FEATURE_CSV_PATH}, 不启用相似度.")

    # 2) 读取 train.parquet
    if (not os.path.exists(TRAIN_PATH)) or (os.path.getsize(TRAIN_PATH) == 0):
        print(f"训练文件 '{TRAIN_PATH}' 不存在或为空, 无法继续.")
        sys.exit(1)
    df_main = pd.read_parquet(TRAIN_PATH)
    del df_main['partition_id']

    categorical_features = ['feature_09', 'feature_10', 'feature_11']
    for feature in categorical_features:
        df_main[feature] = df_main[feature].astype('category').cat.codes

    # 3.1) 检查并转换所有其他类别类型的列（如果有）
    other_categorical = df_main.select_dtypes(['category']).columns.tolist()
    # 移除已经编码的 categorical_features
    other_categorical = [col for col in other_categorical if col not in categorical_features]
    for feature in other_categorical:
        df_main[feature] = df_main[feature].astype('category').cat.codes
    if other_categorical:
        print(f"已编码其他类别列: {other_categorical}")

    # 4) 内存优化前，先填充缺失值
    # 仅对数值型列填充缺失值
    numeric_cols = df_main.select_dtypes(include=[np.number]).columns
    df_main[numeric_cols] = df_main[numeric_cols].fillna(-3.0)

    df_main = reduce_mem_usage(df_main, float16_as32=False)

    # 3) 手动构建 responder_X lag(1)
    lag_cols = [f"responder_{i}" for i in range(9)]
    df_main.sort_values(['symbol_id','date_id','time_id'], inplace=True, ignore_index=True)
    for i in range(9):
        base_col = f"responder_{i}"
        new_col = f"responder_{i}_lag_1"
        df_main[new_col] = df_main.groupby('symbol_id')[base_col].shift(1)
    new_lag_cols = [f"responder_{i}_lag_1" for i in range(9)]
    df_main.dropna(subset=new_lag_cols, inplace=True)
    df_main.reset_index(drop=True, inplace=True)
    print("已构建lag列, shape:", df_main.shape)

    # 4) 跳过天数 SKIP_DATES
    df_main = df_main[df_main['date_id'] >= SKIP_DATES].reset_index(drop=True)
    print(f"过滤后 shape={df_main.shape}")

    # 5) 多次Episode, 进行 MCTS 搜索表达式
    final_exprs = train_mcts_feature_engineer(
        df_main,
        target_col='responder_6',
        unary_ops=['log','exp','sqrt','square'],
        binary_ops=['+','-','*','/'],
        feature_list=None,
        max_depth=5,
        num_episodes=1,
        n_sim=5,
        max_steps_per_episode=5,
        sim_matrix=sim_matrix,
        use_similarity=use_similarity
    )

    print("=== MCTS搜索完成 ===")
    for i, expr in enumerate(final_exprs):
        print(f"Episode {i+1}: Expression => {expr}")
