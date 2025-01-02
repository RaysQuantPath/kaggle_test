import os
import sys
import json
import numpy as np
import pandas as pd
import random
from collections import defaultdict
from sklearn.model_selection import KFold
from catboost import CatBoostRegressor

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F

##############################################################################
# ---------------------------- 全局配置 --------------------------------------
##############################################################################
ROOT_DIR = r'/well/ludwig/users/mil024/YY/'
TRAIN_PATH = os.path.join(ROOT_DIR, 'train.parquet')

os.makedirs(ROOT_DIR, exist_ok=True)

TRAINING = True
FEATURE_NAMES_XLSTM = [f"feature_{i:02d}" for i in range(79)]
FEATURE_NAMES_OTHER = [f"feature_{i:02d}" for i in range(79)]
NUM_VALID_DATES = 200
NUM_TEST_DATES = 120
SKIP_DATES = 500
N_FOLD = 5
SEQUENCE_LENGTH = 5

# 是否多GPU
USE_MULTI_GPU = True
GPU_DEVICE_IDS = [0, 1, 2, 3]

# 让新特征与已有特征“正交”惩罚系数
ORTH_LAMBDA = 0.05

# 请修改为您自己features.csv的路径(若要用标签相似度)
FEATURE_CSV_PATH = '/well/ludwig/users/mil024/YY/features.csv'


##############################################################################
# ----------------------  内存优化函数  --------------------------------------
##############################################################################
def reduce_mem_usage(df, float16_as32=True):
    start_mem = df.memory_usage().sum() / 1024 ** 2
    print('Memory usage of dataframe is {:.2f} MB'.format(start_mem))
    for col in df.columns:
        col_type = df[col].dtype
        if col_type != object and str(col_type) != 'category':
            c_min, c_max = df[col].min(), df[col].max()
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
    end_mem = df.memory_usage().sum() / 1024 ** 2
    print('Memory usage after optimization is: {:.2f} MB'.format(end_mem))
    print('Decreased by {:.1f}%'.format(100 * (start_mem - end_mem) / start_mem))
    return df


##############################################################################
# ----------------  (可选) 标签相似度矩阵的辅助函数  -------------------------
##############################################################################
def compute_tag_similarity_matrix(tag_df):
    feats = tag_df.index.tolist()
    arr = tag_df.astype(int).values
    sim_matrix = {}
    for i, f1 in enumerate(feats):
        sim_matrix[f1] = {}
        for j, f2 in enumerate(feats):
            overlap = np.sum(arr[i] & arr[j])
            sim_matrix[f1][f2] = float(overlap)
    return sim_matrix


##############################################################################
# ----------------  算子树 (OperatorTree) 与 JSON序列化  --------------------
##############################################################################
class OperatorTree:
    def __init__(self, operator=None, children=None, depth=1):
        self.operator = operator
        self.children = children if children else []
        self.depth = depth

    def __repr__(self):
        return f"OpTree(operator={self.operator}, depth={self.depth}, children={self.children})"


def operator_tree_to_dict(tree):
    if tree.operator is None and len(tree.children) == 1 and isinstance(tree.children[0], str):
        return {
            "type": "leaf",
            "operator": None,
            "children": tree.children,
            "depth": tree.depth
        }
    else:
        child_dicts = []
        for c in tree.children:
            if isinstance(c, OperatorTree):
                child_dicts.append(operator_tree_to_dict(c))
            elif isinstance(c, str):
                child_dicts.append({
                    "type": "leaf",
                    "operator": None,
                    "children": [c],
                    "depth": 1
                })
            else:
                raise ValueError("Unknown child in operator_tree_to_dict.")
        return {
            "type": "node",
            "operator": tree.operator,
            "children": child_dicts,
            "depth": tree.depth
        }


def dict_to_operator_tree(d):
    node_type = d["type"]
    depth = d["depth"]
    operator = d["operator"]
    if node_type == "leaf":
        return OperatorTree(operator=None, children=d["children"], depth=depth)
    else:
        child_trees = []
        for cdict in d["children"]:
            child_trees.append(dict_to_operator_tree(cdict))
        return OperatorTree(operator=operator, children=child_trees, depth=depth)


##############################################################################
# -------------- 关键：补充 evaluate_operator_tree 函数  ---------------------
##############################################################################
def evaluate_operator_tree(tree, df):
    """
    递归地在 DataFrame 上计算算子树对应的新特征值。
    df: 包含原始 feature_列 以及responder列(如果存在)。
    tree: OperatorTree 对象。

    返回: shape=(len(df),) 的 np.array
    """
    op = tree.operator
    children = tree.children

    # 叶子节点：operator=None, children=['feature_00'] 或类似
    if op is None and len(children) == 1 and isinstance(children[0], str):
        # 直接返回 df 对应列
        return df[children[0]].values

    # 否则，需要先递归计算子节点
    child_vals = []
    for c in children:
        if isinstance(c, OperatorTree):
            child_vals.append(evaluate_operator_tree(c, df))
        elif isinstance(c, str):
            child_vals.append(df[c].values)
        else:
            raise ValueError("Unknown child type in operator tree.")

    if op in ['+', '-', '*', '/']:
        if len(child_vals) != 2:
            raise ValueError(f"Binary operator '{op}'需要2个子节点.")
        c1, c2 = child_vals
        if op == '+':
            return c1 + c2
        elif op == '-':
            return c1 - c2
        elif op == '*':
            return c1 * c2
        elif op == '/':
            return np.where(np.abs(c2) < 1e-9, 0, c1 / (c2 + 1e-9))
    else:
        # 单目运算
        if len(child_vals) != 1:
            raise ValueError(f"Unary operator '{op}'需要1个子节点.")
        c = child_vals[0]
        if op == 'log':
            return np.log(np.abs(c) + 1e-9)
        elif op == 'exp':
            return np.exp(c)
        elif op == 'sqrt':
            return np.sqrt(np.abs(c))
        elif op == 'square':
            return c ** 2
        else:
            raise NotImplementedError(f"Unsupported operator: {op}")


##############################################################################
# ------------- 评估函数 / CatBoost / Weighted R² / DRL 逻辑 ---------------
##############################################################################
def weighted_r2_score(y_true, y_pred, w):
    numerator = np.sum(w * (y_true - y_pred) ** 2)
    denominator = np.sum(w * (y_true ** 2)) + 1e-15
    return 1 - numerator / denominator


def evaluate_with_cv(X, y, w, model_fn, n_splits=5, random_state=42):
    from sklearn.model_selection import KFold
    kf = KFold(n_splits=n_splits, shuffle=True, random_state=random_state)
    scores = []
    for train_idx, val_idx in kf.split(X):
        X_train, X_val = X[train_idx], X[val_idx]
        y_train, y_val = y[train_idx], y[val_idx]
        w_train, w_val = w[train_idx], w[val_idx]
        model = model_fn()
        model.fit(X_train, y_train, sample_weight=w_train,
                  eval_set=(X_val, y_val), use_best_model=False, verbose=False)

        y_pred = model.predict(X_val)
        score = weighted_r2_score(y_val, y_pred, w_val)
        scores.append(score)
    return np.mean(scores)


def default_model_fn():
    return CatBoostRegressor(
        iterations=2000,
        depth=6,
        learning_rate=0.01,
        task_type='GPU',
        devices='0,1,2,3',
        verbose=0,
        early_stopping_rounds=200
    )


ORTH_LAMBDA = 0.05


def compute_orthogonality_penalty(new_col, X_existing):
    if X_existing.shape[1] == 0:
        return 0.0
    new_centered = new_col - new_col.mean()
    denom_new = new_centered.std() + 1e-15
    max_corr_abs = 0.0
    for j in range(X_existing.shape[1]):
        col_j = X_existing[:, j]
        col_j_centered = col_j - col_j.mean()
        denom_j = col_j_centered.std() + 1e-15
        corr = (new_centered * col_j_centered).mean() / (denom_new * denom_j)
        if abs(corr) > max_corr_abs:
            max_corr_abs = abs(corr)
    penalty = ORTH_LAMBDA * max_corr_abs
    return penalty


def collect_leaf_features(tree):
    leaves = set()

    def dfs(t):
        if t.operator is None and len(t.children) == 1 and isinstance(t.children[0], str):
            leaves.add(t.children[0])
        else:
            for c in t.children:
                if isinstance(c, OperatorTree):
                    dfs(c)
                elif isinstance(c, str):
                    leaves.add(c)

    dfs(tree)
    return leaves


##############################################################################
# -----------------  环境: FeatureEngineeringEnv  ----------------------------
##############################################################################
class FeatureEngineeringEnv:
    def __init__(
            self,
            df,
            target_col='responder_6',
            weight_col=None,
            unary_ops=['log', 'exp', 'sqrt', 'square'],
            binary_ops=['+', '-', '*', '/'],
            max_depth=2,
            model_fn=default_model_fn,
            n_splits=3,
            random_state=42,
            sim_matrix=None,
            use_similarity=False
    ):
        self.df_original = df
        self.target_col = target_col

        self.y = df[target_col].values
        if weight_col:
            self.w = df[weight_col].values
        else:
            self.w = np.ones(len(df), dtype=np.float32)

        self.unary_ops = unary_ops
        self.binary_ops = binary_ops
        self.max_depth = max_depth
        self.model_fn = model_fn
        self.n_splits = n_splits
        self.random_state = random_state

        all_cols = [c for c in df.columns if c.startswith("feature_")]
        self.init_features = all_cols

        self.operator_trees = []
        for col in self.init_features:
            t = OperatorTree(operator=None, children=[col], depth=1)
            self.operator_trees.append(t)

        self.X_current = df[self.init_features].values
        self.current_score = evaluate_with_cv(
            self.X_current, self.y, self.w, self.model_fn, self.n_splits, self.random_state
        )

        self.sim_matrix = sim_matrix
        self.use_similarity = use_similarity
        self.step_count = 0

    def reset(self):
        self.step_count = 0
        self.operator_trees = []
        for col in self.init_features:
            t = OperatorTree(operator=None, children=[col], depth=1)
            self.operator_trees.append(t)
        self.X_current = self.df_original[self.init_features].values
        self.current_score = evaluate_with_cv(
            self.X_current, self.y, self.w, self.model_fn, self.n_splits, self.random_state
        )
        return self._get_state()

    def _get_state(self):
        return np.array([len(self.operator_trees), self.current_score], dtype=np.float32)

    def step(self, new_tree):
        self.step_count += 1
        # 1) 深度检查
        if new_tree.depth > self.max_depth:
            return self._get_state(), -0.01, False, {}

        # 2) 重复检查
        new_json = json.dumps(operator_tree_to_dict(new_tree))
        exist_jsons = [json.dumps(operator_tree_to_dict(t)) for t in self.operator_trees]
        if new_json in exist_jsons:
            return self._get_state(), -0.005, False, {}

        # 3) 评估 => evaluate_operator_tree
        new_col = evaluate_operator_tree(new_tree, self.df_original)

        # 正交惩罚
        penalty_orth = compute_orthogonality_penalty(new_col, self.X_current)

        X_new = np.column_stack([self.X_current, new_col])
        new_score = evaluate_with_cv(
            X_new, self.y, self.w, self.model_fn, self.n_splits, self.random_state
        )

        delta = new_score - self.current_score
        reward = delta - penalty_orth

        self.operator_trees.append(new_tree)
        self.X_current = X_new
        self.current_score = new_score
        return self._get_state(), reward, False, {}

    def sample_random_tree(self):
        if random.random() < 0.5:
            return self._sample_random_tree_unary()
        else:
            return self._sample_random_tree_binary()

    def _sample_random_tree_unary(self):
        op = random.choice(self.unary_ops)
        child = random.choice(self.operator_trees)
        return OperatorTree(operator=op, children=[child], depth=child.depth + 1)

    def _sample_random_tree_binary(self):
        op = random.choice(self.binary_ops)
        if (not self.use_similarity) or (self.sim_matrix is None):
            t1 = random.choice(self.operator_trees)
            t2 = random.choice(self.operator_trees)
            d = max(t1.depth, t2.depth) + 1
            return OperatorTree(operator=op, children=[t1, t2], depth=d)
        else:
            trees = self.operator_trees
            if len(trees) < 2:
                return self._sample_random_tree_unary()

            pair_candidates = []
            weights = []
            for i in range(len(trees)):
                for j in range(i + 1, len(trees)):
                    t1 = trees[i]
                    t2 = trees[j]
                    leaves1 = collect_leaf_features(t1)
                    leaves2 = collect_leaf_features(t2)
                    if len(leaves1) == 0 or len(leaves2) == 0:
                        overlap_value = 0.0
                    else:
                        sum_ov = 0.0
                        count_ov = 0
                        for lf1 in leaves1:
                            for lf2 in leaves2:
                                ov = self.sim_matrix.get(lf1, {}).get(lf2, 0.0)
                                sum_ov += ov
                                count_ov += 1
                        overlap_value = sum_ov / max(count_ov, 1)
                    pair_candidates.append((t1, t2))
                    w_ij = np.exp(overlap_value)
                    weights.append(w_ij)

            w_arr = np.array(weights, dtype=float)
            if w_arr.sum() < 1e-9:
                t1 = random.choice(trees)
                t2 = random.choice(trees)
                d = max(t1.depth, t2.depth) + 1
                return OperatorTree(operator=op, children=[t1, t2], depth=d)
            probs = w_arr / w_arr.sum()
            chosen_idx = np.random.choice(len(pair_candidates), p=probs)
            t1, t2 = pair_candidates[chosen_idx]
            d = max(t1.depth, t2.depth) + 1
            return OperatorTree(operator=op, children=[t1, t2], depth=d)


##############################################################################
# ----------------- ReplayBuffer, DQN, DoubleDQNAgent ------------------------
##############################################################################
class ReplayBuffer:
    def __init__(self, capacity=10000):
        self.capacity = capacity
        self.memory = []
        self.position = 0

    def push(self, s, action_tree_json, r, s_next, done):
        if len(self.memory) < self.capacity:
            self.memory.append(None)
        self.memory[self.position] = (s, action_tree_json, r, s_next, done)
        self.position = (self.position + 1) % self.capacity

    def sample(self, batch_size):
        return random.sample(self.memory, batch_size)

    def __len__(self):
        return len(self.memory)


def embed_state_and_action(state, tree):
    stats = defaultdict(int)

    def traverse(t):
        if t.operator is None and len(t.children) == 1 and isinstance(t.children[0], str):
            stats['leaf_count'] += 1
            stats['max_depth'] = max(stats['max_depth'], t.depth)
        else:
            op = t.operator
            stats['max_depth'] = max(stats['max_depth'], t.depth)
            if op in ['log', 'exp', 'sqrt', 'square']:
                stats['unary_count'] += 1
            elif op in ['+', '-', '*', '/']:
                stats['binary_count'] += 1
            if op == 'log':
                stats['op_log'] += 1
            elif op == 'exp':
                stats['op_exp'] += 1
            elif op == 'sqrt':
                stats['op_sqrt'] += 1
            elif op == 'square':
                stats['op_square'] += 1
            elif op == '+':
                stats['op_plus'] += 1
            elif op == '-':
                stats['op_minus'] += 1
            elif op == '*':
                stats['op_mul'] += 1
            elif op == '/':
                stats['op_div'] += 1
            for c in t.children:
                if isinstance(c, OperatorTree):
                    traverse(c)
                elif isinstance(c, str):
                    stats['leaf_count'] += 1

    for opk in ['unary_count', 'binary_count', 'max_depth', 'leaf_count',
                'op_log', 'op_exp', 'op_sqrt', 'op_square', 'op_plus', 'op_minus', 'op_mul', 'op_div']:
        stats[opk] = 0
    traverse(tree)

    emb = np.array([
        state[0],
        state[1],
        stats['unary_count'],
        stats['binary_count'],
        stats['max_depth'],
        stats['leaf_count'],
        stats['op_log'],
        stats['op_exp'],
        stats['op_sqrt'],
        stats['op_square'],
        stats['op_plus'],
        stats['op_minus'],
        stats['op_mul'],
        stats['op_div']
    ], dtype=np.float32)
    return emb


class DQNetwork(nn.Module):
    def __init__(self, in_dim=14, hidden_dim=64):
        super(DQNetwork, self).__init__()
        self.fc1 = nn.Linear(in_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.out = nn.Linear(hidden_dim, 1)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        return self.out(x)


class DoubleDQNAgent:
    def __init__(self, in_dim=14, hidden_dim=64, replay_capacity=10000, batch_size=32,
                 gamma=0.99, lr=1e-3, epsilon_start=1.0, epsilon_min=0.1, epsilon_decay=0.995):
        self.batch_size = batch_size
        self.gamma = gamma

        base_online = DQNetwork(in_dim, hidden_dim)
        base_target = DQNetwork(in_dim, hidden_dim)

        if USE_MULTI_GPU:
            self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
            self.online_net = nn.DataParallel(base_online, device_ids=GPU_DEVICE_IDS).cuda()
            self.target_net = nn.DataParallel(base_target, device_ids=GPU_DEVICE_IDS).cuda()
        else:
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            self.online_net = base_online.to(self.device)
            self.target_net = base_target.to(self.device)

        self.target_net.load_state_dict(self.online_net.state_dict())
        self.target_net.eval()
        self.optimizer = optim.Adam(self.online_net.parameters(), lr=lr)
        self.replay_buffer = ReplayBuffer(replay_capacity)

        self.epsilon = epsilon_start
        self.epsilon_min = epsilon_min
        self.epsilon_decay = epsilon_decay

    def select_action(self, state, env):
        N_CANDIDATES = 10
        candidates = [env.sample_random_tree() for _ in range(N_CANDIDATES)]
        if random.random() < self.epsilon:
            return random.choice(candidates)
        else:
            best_tree = None
            best_q = -1e9
            for c in candidates:
                sa_emb = embed_state_and_action(state, c)
                sa_t = torch.tensor(sa_emb, dtype=torch.float32).to(self.device).unsqueeze(0)
                with torch.no_grad():
                    q_val = self.online_net(sa_t).item()
                if q_val > best_q:
                    best_q = q_val
                    best_tree = c
            return best_tree

    def store_transition(self, s, action_tree, r, s_next, done):
        dict_obj = operator_tree_to_dict(action_tree)
        action_json = json.dumps(dict_obj)
        self.replay_buffer.push(s, action_json, r, s_next, done)

    def update_epsilon(self):
        self.epsilon = max(self.epsilon_min, self.epsilon * self.epsilon_decay)

    def train_step(self, env):
        if len(self.replay_buffer) < self.batch_size:
            return
        transitions = self.replay_buffer.sample(self.batch_size)
        batch_s, batch_a_json, batch_r, batch_s_next, batch_done = zip(*transitions)
        batch_s = np.stack(batch_s)
        batch_r = np.array(batch_r).reshape(-1, 1)
        batch_s_next = np.stack(batch_s_next)
        batch_done = np.array(batch_done).reshape(-1, 1)

        sa_emb_list = []
        for i in range(self.batch_size):
            s_arr = batch_s[i]
            a_obj = json.loads(batch_a_json[i])
            a_tree = dict_to_operator_tree(a_obj)
            sa_emb = embed_state_and_action(s_arr, a_tree)
            sa_emb_list.append(sa_emb)
        sa_emb_batch = torch.tensor(sa_emb_list, dtype=torch.float32).to(self.device)
        q_sa = self.online_net(sa_emb_batch)

        with torch.no_grad():
            batch_best_q_next = []
            for i in range(self.batch_size):
                if batch_done[i]:
                    batch_best_q_next.append(0.0)
                    continue
                s_next_arr = batch_s_next[i]
                N_CAND2 = 10
                cands2 = [env.sample_random_tree() for _ in range(N_CAND2)]
                best_qv = -1e9
                best_tree = None
                for c2 in cands2:
                    emb2 = embed_state_and_action(s_next_arr, c2)
                    emb2_t = torch.tensor(emb2, dtype=torch.float32).to(self.device).unsqueeze(0)
                    qv = self.online_net(emb2_t).item()
                    if qv > best_qv:
                        best_qv = qv
                        best_tree = c2
                if best_tree is None:
                    batch_best_q_next.append(0.0)
                else:
                    emb3 = embed_state_and_action(s_next_arr, best_tree)
                    emb3_t = torch.tensor(emb3, dtype=torch.float32).to(self.device).unsqueeze(0)
                    qv2 = self.target_net(emb3_t).item()
                    batch_best_q_next.append(qv2)
            batch_best_q_next = np.array(batch_best_q_next).reshape(-1, 1)
            y = batch_r + self.gamma * (1 - batch_done) * batch_best_q_next

        y_t = torch.tensor(y, dtype=torch.float32).to(self.device)
        loss = F.mse_loss(q_sa, y_t)
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

    def update_target(self):
        self.target_net.load_state_dict(self.online_net.state_dict())


##############################################################################
# --------------  训练流程: DRL自动造特征  ----------------------------------
##############################################################################
def train_dqn_feature_engineer(
        df,
        target_col='responder_6',
        weight_col=None,
        num_episodes=5,
        max_steps_per_ep=5,
        max_depth=2,
        k_splits=3,
        sim_matrix=None,
        use_similarity=False
):
    env = FeatureEngineeringEnv(
        df=df,
        target_col=target_col,
        weight_col=weight_col,
        unary_ops=['log', 'exp', 'sqrt', 'square'],
        binary_ops=['+', '-', '*', '/'],
        max_depth=max_depth,
        model_fn=default_model_fn,
        n_splits=k_splits,
        random_state=42,
        sim_matrix=sim_matrix,
        use_similarity=use_similarity
    )
    agent = DoubleDQNAgent(
        in_dim=14,
        hidden_dim=64,
        replay_capacity=10000,
        batch_size=32,
        gamma=0.99,
        lr=1e-3,
        epsilon_start=1.0,
        epsilon_min=0.1,
        epsilon_decay=0.995
    )

    rewards_history = []
    for ep in range(num_episodes):
        state = env.reset()
        ep_reward = 0.0
        for t in range(max_steps_per_ep):
            action_tree = agent.select_action(state, env)
            next_state, reward, done, _ = env.step(action_tree)
            agent.store_transition(state, action_tree, reward, next_state, done)
            agent.train_step(env)
            state = next_state
            ep_reward += reward
            if done:
                break
        agent.update_epsilon()
        agent.update_target()

        print(f"Episode {ep + 1}/{num_episodes}, Reward={ep_reward:.4f}, Eps={agent.epsilon:.3f}")
        rewards_history.append(ep_reward)
    print("===== DRL 造特征结束 =====")
    print("最终Score:", env.current_score)
    print("特征个数:", len(env.operator_trees))
    return env, agent, rewards_history


##############################################################################
# ---------------------------- 主入口 ----------------------------------------
##############################################################################
if TRAINING:
    # 1) 若您有 features.csv，用于标签相似度 => 读csv => sim_matrix
    if os.path.exists(FEATURE_CSV_PATH):
        tag_csv = pd.read_csv(FEATURE_CSV_PATH)
        tag_csv.set_index('feature', inplace=True)
        for c in tag_csv.columns:
            tag_csv[c] = tag_csv[c].astype(bool)
        sim_matrix = compute_tag_similarity_matrix(tag_csv)
        use_similarity = True
        print("已读取 features.csv 并计算标签相似度矩阵.")
    else:
        sim_matrix = None
        use_similarity = False
        print(f"未找到 {FEATURE_CSV_PATH}, 不启用标签相似度.")

    # 2) 读取 train.parquet
    if os.path.getsize(TRAIN_PATH) > 0:
        df_main = pd.read_parquet(TRAIN_PATH)
        df_main.fillna(3.0, inplace=True)
        df_main = reduce_mem_usage(df_main, float16_as32=False)
    else:
        print(f"训练文件 '{TRAIN_PATH}' 为空，无法继续。")
        sys.exit(1)

    # 3) 手动构建 responder_X lag(1)
    lag_cols = [f"responder_{i}" for i in range(9)]
    df_main.sort_values(['symbol_id', 'date_id', 'time_id'], inplace=True, ignore_index=True)
    for i in range(9):
        base_col = f"responder_{i}"
        new_col = f"responder_{i}_lag_1"
        df_main[new_col] = df_main.groupby('symbol_id')[base_col].shift(1)

    # dropna
    new_lag_cols = [f"responder_{i}_lag_1" for i in range(9)]
    df_main.dropna(subset=new_lag_cols, inplace=True)
    df_main.reset_index(drop=True, inplace=True)

    print("已手动构建 lag，并重命名, 当前shape:", df_main.shape)

    # 4) date_id >= SKIP_DATES
    df_main = df_main[df_main['date_id'] >= SKIP_DATES].reset_index(drop=True)

    # 5) train/valid/test
    dates = df_main['date_id'].unique()
    test_dates = dates[-NUM_TEST_DATES:]
    remain_dates = dates[:-NUM_TEST_DATES]
    valid_dates = remain_dates[-NUM_VALID_DATES:] if NUM_VALID_DATES > 0 else []
    train_dates = remain_dates[:-NUM_VALID_DATES] if NUM_VALID_DATES > 0 else remain_dates

    # 6) symbol_id => int
    symbol_ids = df_main['symbol_id'].unique()
    sid2idx = {s: i for i, s in enumerate(symbol_ids)}
    df_main['symbol_id'] = df_main['symbol_id'].map(sid2idx)

    print("数据读取与构建lag完成, shape:", df_main.shape)

    # 7) 选train集 => DRL
    df_train = df_main[df_main['date_id'].isin(train_dates)].reset_index(drop=True)

    env, agent, rewards_hist = train_dqn_feature_engineer(
        df=df_train,
        target_col='responder_6',
        weight_col=None,
        num_episodes=20,
        max_steps_per_ep=5,
        max_depth=10,
        k_splits=3,
        sim_matrix=sim_matrix,
        use_similarity=use_similarity
    )

    print("=== 训练完成 ===")
    print("最终环境score:", env.current_score)
    print("新特征总数:", len(env.operator_trees))
    print("所有特征树:")
    for t in env.operator_trees:
        print("  ", t)
