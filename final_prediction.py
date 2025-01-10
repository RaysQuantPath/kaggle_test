import os
import joblib
import numpy as np
import pandas as pd
import random
import warnings
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import copy

from iTransformer import iTransformer  # 这里是你原本的 import
# 但是我们在下面直接重写了 iTransformer 类（带因果掩码 + 全序列输出）

# CatBoost / XGBoost / sklearn
import catboost as cbt
import xgboost as xgb
from sklearn.linear_model import Ridge
from sklearn.model_selection import KFold

import gc
from glob import glob
import pickle

warnings.filterwarnings("ignore")

###########################################################################
# 全局路径 & 参数
###########################################################################
ROOT_DIR = r'/well/ludwig/users/mil024/YY/'
TRAIN_PATH = os.path.join(ROOT_DIR, 'train.parquet')
DAY_SPLIT_DIR = os.path.join(ROOT_DIR, 'daily_data_final')  # 分日数据存放目录

MODEL_DIR = ROOT_DIR + 'final_pred_666'
MODEL_PATH = ROOT_DIR + 'pretrained_models_final_666'
os.makedirs(ROOT_DIR, exist_ok=True)
os.makedirs(MODEL_DIR, exist_ok=True)
os.makedirs(MODEL_PATH, exist_ok=True)

TRAINING = True

NUM_VALID_DATES = 169
NUM_TEST_DATES = 340
SKIP_DATES = 510

LOOKBACK_LEN = 17
TARGET_COL = 'responder_6'
WEIGHT_COL = 'weight'
DATE_COL = 'date_id'
device_id = '0,1,2,3'
device_id_list = [0, 1, 2, 3]
tree_iteration_num = 2000
online_tree_iteration_num = 600
epoch_long = 10
epoch_short = 5

# 这几个列是类别列，需要 CatBoost 来自动处理
CATEGORICAL_COLS = ['feature_09', 'feature_10', 'feature_11']

EXCLUDE_COLS = [
    TARGET_COL, WEIGHT_COL, DATE_COL, 'symbol_id', 'partition_id',
    'responder_0', 'responder_1', 'responder_2', 'responder_3',
    'responder_4', 'responder_5', 'responder_7', 'responder_8'
]

DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
ONLINE_UPDATE_FREQ = 17


###########################################################################
# 简单特征工程函数(可自定义)
###########################################################################
def feature_generation(df):
    """
    根据题目中给出的 195 条 gen_i => OpTree(...)，逐行生成特征。
    - 对 log(...) => log(abs(...)+1e-9)
    - 对 sqrt(...) => sqrt(abs(...)+1e-9)
    - 对 exp(...) => exp(...)
    - 若是纯数学恒等 => 跳过
    - 若表达式重复 => 跳过
    """
    # 已知表达式集合：如果某条跟之前一模一样，就不再生成
    known_expressions = set()

    def add_column(gen_name, expr_str):
        print(f"=> add_column {gen_name}: {expr_str}")
        if expr_str not in known_expressions:
            # 直接用 df.eval(..., inplace=True)，使用默认引擎 (numexpr)，去掉 np. 前缀
            df.eval(f"{gen_name} = {expr_str}", inplace=True)
            known_expressions.add(expr_str)
        else:
            pass

    # ========== 以下按照题目给出的顺序，一行行写死 ==========

    # gen_0 => log(feature_45)
    add_column("gen_0", "log(abs(feature_45) + 1e-9)")

    # gen_1 => 数学等价于 feature_45 => 跳过

    # gen_2 => sqrt(feature_45)
    add_column("gen_2", "sqrt(abs(feature_45) + 1e-9)")

    # gen_3 => log(feature_45) => 与 gen_0 重复 => 跳过

    # gen_4 => feature_45 => 跳过

    # gen_5 => sqrt(log(feature_45))
    add_column("gen_5", "sqrt(abs(log(abs(feature_45) + 1e-9)) + 1e-9)")

    # gen_6 => log(log(feature_45))
    add_column("gen_6", "log(abs(log(abs(feature_45) + 1e-9)) + 1e-9)")

    # gen_7 => exp(feature_59)
    add_column("gen_7", "exp(feature_59)")

    # gen_8 => sqrt(feature_16)
    add_column("gen_8", "sqrt(abs(feature_16) + 1e-9)")

    # gen_9 => log(feature_40)
    add_column("gen_9", "log(abs(feature_40) + 1e-9)")

    # gen_10 => exp(feature_25)
    add_column("gen_10", "exp(feature_25)")

    # gen_11 => sqrt(feature_46)
    add_column("gen_11", "sqrt(abs(feature_46) + 1e-9)")

    # gen_12 => 与 gen_6 重复 => 跳过

    # gen_13 => exp(feature_46)
    add_column("gen_13", "exp(feature_46)")

    # gen_14 => sqrt(feature_59)
    add_column("gen_14", "sqrt(abs(feature_59) + 1e-9)")

    # gen_15 => log(sqrt(log(feature_45)))
    add_column("gen_15", "log(abs(sqrt(abs(log(abs(feature_45) + 1e-9))) + 1e-9) + 1e-9)")

    # gen_16 => exp(log(log(feature_45)))
    add_column("gen_16", "exp(log(abs(log(abs(feature_45) + 1e-9)) + 1e-9))")

    # gen_17 => sqrt(feature_36)
    add_column("gen_17", "sqrt(abs(feature_36) + 1e-9)")

    # gen_18 => log(feature_25)
    add_column("gen_18", "log(abs(feature_25) + 1e-9)")

    # gen_19 => exp(sqrt(feature_46))
    add_column("gen_19", "exp(sqrt(abs(feature_46) + 1e-9))")

    # gen_20 => sqrt(sqrt(feature_59))
    add_column("gen_20", "sqrt(abs(sqrt(abs(feature_59) + 1e-9)) + 1e-9)")

    # gen_21 => log(sqrt(feature_45))
    add_column("gen_21", "log(abs(sqrt(abs(feature_45) + 1e-9)) + 1e-9)")

    # gen_22 => exp(feature_07)
    add_column("gen_22", "exp(feature_07)")

    # gen_23 => sqrt(feature_06)
    add_column("gen_23", "sqrt(abs(feature_06) + 1e-9)")

    # gen_24 => log(feature_49)
    add_column("gen_24", "log(abs(feature_49) + 1e-9)")

    # gen_25 => exp(feature_29)
    add_column("gen_25", "exp(feature_29)")

    # gen_26 => sqrt(exp(feature_25))
    add_column("gen_26", "sqrt(abs(exp(feature_25)) + 1e-9)")

    # gen_27 => log(feature_22)
    add_column("gen_27", "log(abs(feature_22) + 1e-9)")

    # gen_28 => exp(feature_38)
    add_column("gen_28", "exp(feature_38)")

    # gen_29 => 与 gen_5 相同 => 跳过

    # gen_30 => log(sqrt(sqrt(feature_59)))
    add_column("gen_30", "log(abs(sqrt(abs(sqrt(abs(feature_59) + 1e-9)) + 1e-9)) + 1e-9)")

    # gen_31 => exp(feature_54)
    add_column("gen_31", "exp(feature_54)")

    # gen_32 => sqrt(log(sqrt(sqrt(feature_59))))
    add_column("gen_32", "sqrt(abs(log(abs(sqrt(abs(sqrt(abs(feature_59) + 1e-9)) + 1e-9)) + 1e-9)) + 1e-9)")

    # gen_33 => 同 gen_6 => 跳过

    # gen_34 => exp(feature_31)
    add_column("gen_34", "exp(feature_31)")

    # gen_35 => sqrt(sqrt(log(feature_45)))
    add_column("gen_35", "sqrt(abs(sqrt(abs(log(abs(feature_45) + 1e-9))) + 1e-9) + 1e-9)")

    # gen_36 => log(feature_47)
    add_column("gen_36", "log(abs(feature_47) + 1e-9)")

    # gen_37 => 跳过

    # gen_38 => sqrt(sqrt(feature_36))
    add_column("gen_38", "sqrt(abs(sqrt(abs(feature_36) + 1e-9)) + 1e-9)")

    # gen_39 => 跳过(重复)

    # gen_40 => exp(exp(feature_46))
    add_column("gen_40", "exp(exp(feature_46))")

    # gen_41 => sqrt(exp(feature_31))
    add_column("gen_41", "sqrt(abs(exp(feature_31)) + 1e-9)")

    # gen_42 => 跳过

    # gen_43 => exp(sqrt(feature_06))
    add_column("gen_43", "exp(sqrt(abs(feature_06) + 1e-9))")

    # gen_44 => sqrt(feature_03)
    add_column("gen_44", "sqrt(abs(feature_03) + 1e-9)")

    # gen_45 => log(log(feature_49))
    add_column("gen_45", "log(abs(log(abs(feature_49) + 1e-9)) + 1e-9)")

    # gen_47 => log(feature_62)
    add_column("gen_47", "log(abs(feature_62) + 1e-9)")

    # gen_48 => 同 feature_62 => 跳过

    # gen_49 => sqrt(gen_24)
    add_column("gen_49", "sqrt(abs(gen_24) + 1e-9)")

    # gen_50 => log(log(feature_62))
    add_column("gen_50", "log(abs(log(abs(feature_62) + 1e-9)) + 1e-9)")

    # gen_51 => exp(sqrt(gen_24))
    add_column("gen_51", "exp(sqrt(abs(gen_24) + 1e-9))")

    # gen_52 => sqrt(sqrt(gen_24))
    add_column("gen_52", "sqrt(abs(sqrt(abs(gen_24) + 1e-9)) + 1e-9)")

    # gen_53 => 同 gen_47 => 跳过

    # gen_54 => exp(feature_48)
    add_column("gen_54", "exp(feature_48)")

    # gen_55 => sqrt(feature_62)
    add_column("gen_55", "sqrt(abs(feature_62) + 1e-9)")

    # gen_56 => log(exp(sqrt(gen_24)))
    add_column("gen_56", "log(abs(exp(sqrt(abs(gen_24) + 1e-9))) + 1e-9)")

    # gen_57 => exp(log(log(feature_62)))
    add_column("gen_57", "exp(log(abs(log(abs(feature_62) + 1e-9)) + 1e-9))")

    # gen_58 => sqrt(sqrt(feature_62))
    add_column("gen_58", "sqrt(abs(sqrt(abs(feature_62) + 1e-9)) + 1e-9)")

    # gen_59 => 与 gen_56 相同 => 跳过

    # gen_60 => 同 feature_62 => 跳过

    # gen_61 => sqrt(feature_30)
    add_column("gen_61", "sqrt(abs(feature_30) + 1e-9)")

    # gen_62 => log(gen_28)
    add_column("gen_62", "log(abs(gen_28) + 1e-9)")

    # gen_63 => exp(gen_30)
    add_column("gen_63", "exp(gen_30)")

    # gen_64 => 依赖 gen_3 不存在 => 跳过

    # gen_65 => 同 gen_50 => 跳过

    # gen_66 => exp(gen_41)
    add_column("gen_66", "exp(gen_41)")

    # gen_67 => sqrt(feature_40)
    add_column("gen_67", "sqrt(abs(feature_40) + 1e-9)")

    # gen_68 => log(exp(gen_30))
    add_column("gen_68", "log(abs(exp(gen_30)) + 1e-9)")

    # gen_69 => exp(feature_30)
    add_column("gen_69", "exp(feature_30)")

    # gen_70 => sqrt(feature_41)
    add_column("gen_70", "sqrt(abs(feature_41) + 1e-9)")

    # gen_71 => 跳过

    # gen_72 => exp(exp(feature_30))
    add_column("gen_72", "exp(exp(feature_30))")

    # gen_74 => log(gen_52)
    add_column("gen_74", "log(abs(gen_52) + 1e-9)")

    # gen_75 => exp(feature_07) => 与 gen_22 相同 => 跳过

    # gen_76 => sqrt(gen_38)
    add_column("gen_76", "sqrt(abs(gen_38) + 1e-9)")

    # gen_77 => log(feature_50)
    add_column("gen_77", "log(abs(feature_50) + 1e-9)")

    # gen_78 => exp(gen_6)
    add_column("gen_78", "exp(gen_6)")

    # gen_79 => sqrt(gen_36)
    add_column("gen_79", "sqrt(abs(gen_36) + 1e-9)")

    # gen_80 => log(feature_03)
    add_column("gen_80", "log(abs(feature_03) + 1e-9)")

    # gen_81 => exp(gen_67)
    add_column("gen_81", "exp(gen_67)")

    # gen_82 => sqrt(exp(gen_67))
    add_column("gen_82", "sqrt(abs(exp(gen_67)) + 1e-9)")

    # gen_83 => log(log(feature_03))
    add_column("gen_83", "log(abs(log(abs(feature_03) + 1e-9)) + 1e-9)")

    # gen_84 => exp(feature_59) => 与 gen_7 相同 => 跳过

    # gen_85 => sqrt(log(log(feature_03)))
    add_column("gen_85", "sqrt(abs(log(abs(log(abs(feature_03) + 1e-9)) + 1e-9)) + 1e-9)")

    # gen_86 => log(log(feature_50))
    add_column("gen_86", "log(abs(log(abs(feature_50) + 1e-9)) + 1e-9)")

    # gen_87 => exp(feature_61)
    add_column("gen_87", "exp(feature_61)")

    # gen_88 => sqrt(gen_66)
    add_column("gen_88", "sqrt(abs(gen_66) + 1e-9)")

    # gen_89 => log(exp(gen_6))
    add_column("gen_89", "log(abs(exp(gen_6)) + 1e-9)")

    # gen_90 => exp(gen_7)
    add_column("gen_90", "exp(gen_7)")

    # gen_91 => sqrt(log(feature_50))
    add_column("gen_91", "sqrt(abs(log(abs(feature_50) + 1e-9)) + 1e-9)")

    # gen_92 => log(feature_56)
    add_column("gen_92", "log(abs(feature_56) + 1e-9)")

    # gen_93 => 同 feature_03 => 跳过

    # gen_94 => sqrt(log(gen_52))
    add_column("gen_94", "sqrt(abs(log(abs(gen_52) + 1e-9)) + 1e-9)")

    # gen_95 => 跳过

    # gen_96 => 同 gen_78 => 跳过

    # gen_97 => 跳过

    # gen_98 => log(feature_01)
    add_column("gen_98", "log(abs(feature_01) + 1e-9)")

    # gen_99 => exp(gen_22)
    add_column("gen_99", "exp(gen_22)")

    # gen_100 => sqrt(log(log(feature_50)))
    add_column("gen_100", "sqrt(abs(log(abs(log(abs(feature_50) + 1e-9)) + 1e-9)) + 1e-9)")

    # gen_101 => log(log(log(feature_50)))
    add_column("gen_101", "log(abs(log(abs(log(abs(feature_50) + 1e-9)) + 1e-9)) + 1e-9)")

    # gen_102 => exp(exp(feature_59))
    add_column("gen_102", "exp(exp(feature_59))")

    # gen_103 => 跳过

    # gen_104 => log(sqrt(log(log(log(feature_03)))))
    add_column("gen_104",
               "log(abs(sqrt(abs(log(abs(log(abs(log(abs(feature_03) + 1e-9)) + 1e-9)) + 1e-9)) + 1e-9) ) + 1e-9)")

    # gen_105 => exp(gen_20)
    add_column("gen_105", "exp(gen_20)")

    # gen_106 => sqrt(gen_58)
    add_column("gen_106", "sqrt(abs(gen_58) + 1e-9)")

    # gen_107 => log(exp(feature_59))
    add_column("gen_107", "log(abs(exp(feature_59)) + 1e-9)")

    # gen_108 => exp(feature_27)
    add_column("gen_108", "exp(feature_27)")

    # gen_109 => sqrt(feature_71)
    add_column("gen_109", "sqrt(abs(feature_71) + 1e-9)")

    # gen_110 => log(exp(gen_7))
    add_column("gen_110", "log(abs(exp(gen_7)) + 1e-9)")

    # gen_111 => exp(gen_69)
    add_column("gen_111", "exp(gen_69)")

    # gen_112 => sqrt(feature_03)
    add_column("gen_112", "sqrt(abs(feature_03) + 1e-9)")

    # gen_113 => log(feature_59)
    add_column("gen_113", "log(abs(feature_59) + 1e-9)")

    # gen_114 => exp(gen_41)
    add_column("gen_114", "exp(gen_41)")

    # gen_115 => sqrt(sqrt(gen_66))
    add_column("gen_115", "sqrt(abs(sqrt(abs(gen_66) + 1e-9)) + 1e-9)")

    # gen_116 => 跳过

    # gen_117 => 跳过

    # gen_118 => sqrt(feature_05)
    add_column("gen_118", "sqrt(abs(feature_05) + 1e-9)")

    # gen_119 => log(feature_70)
    add_column("gen_119", "log(abs(feature_70) + 1e-9)")

    # gen_120 => 同 feature_03 => 跳过

    # gen_121 => sqrt(log(sqrt(log(log(feature_03)))))
    add_column("gen_121",
               "sqrt(abs(log(abs(sqrt(abs(log(abs(log(abs(feature_03) + 1e-9)) + 1e-9)) + 1e-9)) + 1e-9)) + 1e-9)")

    # gen_122 => log(sqrt(log(gen_52)))
    add_column("gen_122",
               "log(abs(sqrt(abs(log(abs(gen_52) + 1e-9)) + 1e-9)) + 1e-9)")

    # gen_123 => exp(gen_66)
    add_column("gen_123", "exp(gen_66)")

    # gen_124 => log(gen_76)
    add_column("gen_124", "log(abs(gen_76) + 1e-9)")

    # gen_125 => exp(log(gen_76))
    add_column("gen_125", "exp(log(abs(gen_76) + 1e-9))")

    # gen_126 => sqrt(exp(log(gen_76)))
    add_column("gen_126", "sqrt(abs(gen_76) + 1e-9)")

    # gen_127 => log(sqrt(exp(log(gen_76))))
    add_column("gen_127", "log(abs(sqrt(abs(gen_76) + 1e-9)) + 1e-9)")

    # gen_128 => 跳过

    # gen_129 => sqrt(feature_06) => 与 gen_23 重复 => 跳过

    # gen_130 => log(exp(log(gen_76))) => => log(log(gen_76)) ?
    add_column("gen_130", "log(abs(log(abs(gen_76) + 1e-9)) + 1e-9)")

    # gen_131 => exp(log(gen_76)) => => gen_76 => 跳过

    # gen_132 => 跳过

    # gen_133 => log(sqrt(feature_06))
    add_column("gen_133", "log(abs(gen_23) + 1e-9)")

    # gen_134 => 跳过

    # gen_135 => sqrt(exp(gen_123))
    add_column("gen_135", "sqrt(abs(exp(gen_123)) + 1e-9)")

    # gen_136 => log(log(gen_76))
    add_column("gen_136", "log(abs(log(abs(gen_76) + 1e-9)) + 1e-9)")

    # gen_137 => exp(exp(log(gen_76)))
    add_column("gen_137", "exp(exp(log(abs(gen_76) + 1e-9)))")

    # gen_138 => sqrt(gen_137)
    add_column("gen_138", "sqrt(abs(gen_137) + 1e-9)")

    # gen_139 => 跳过

    # gen_140 => 与 gen_137 相同 => 跳过

    # gen_141 => sqrt(gen_79)
    add_column("gen_141", "sqrt(abs(gen_79) + 1e-9)")

    # gen_142 => log(feature_43)
    add_column("gen_142", "log(abs(feature_43) + 1e-9)")

    # gen_143 => exp(gen_17)
    add_column("gen_143", "exp(gen_17)")

    # gen_144 => sqrt(gen_108)
    add_column("gen_144", "sqrt(abs(gen_108) + 1e-9)")

    # gen_145 => 跳过

    # gen_146 => exp(log(gen_45))
    add_column("gen_146", "exp(log(abs(gen_45) + 1e-9))")

    # gen_147 => sqrt(sqrt(feature_06))
    add_column("gen_147", "sqrt(abs(sqrt(abs(feature_06) + 1e-9)) + 1e-9)")

    # gen_148 => log(feature_46)
    add_column("gen_148", "log(abs(feature_46) + 1e-9)")

    # gen_149 => 跳过

    # gen_150 => sqrt(feature_52)
    add_column("gen_150", "sqrt(abs(feature_52) + 1e-9)")

    # gen_151 => 与 gen_136 一样 => 跳过

    # gen_152 => 跳过

    # gen_153 => 与 gen_88 一样 => 跳过

    # gen_154 => log(gen_78)
    add_column("gen_154", "log(abs(gen_78) + 1e-9)")

    # gen_155 => exp(sqrt(feature_52))
    add_column("gen_155", "exp(sqrt(abs(feature_52) + 1e-9))")

    # gen_156 => sqrt(gen_45)
    add_column("gen_156", "sqrt(abs(gen_45) + 1e-9)")

    # gen_157 => 跳过

    # gen_158 => exp(feature_53)
    add_column("gen_158", "exp(feature_53)")

    # gen_159 => sqrt(log(gen_78))
    add_column("gen_159", "sqrt(abs(log(abs(gen_78) + 1e-9)) + 1e-9)")

    # gen_161 => exp(gen_85)
    add_column("gen_161", "exp(gen_85)")

    # gen_162 => sqrt(gen_41)
    add_column("gen_162", "sqrt(abs(gen_41) + 1e-9)")

    # gen_163 => log(feature_75)
    add_column("gen_163", "log(abs(feature_75) + 1e-9)")

    # gen_164 => 与 gen_43 相同 => 跳过

    # gen_165 => 跳过

    # gen_166 => log(gen_51)
    add_column("gen_166", "log(abs(gen_51) + 1e-9)")

    # gen_167 => exp(feature_37)
    add_column("gen_167", "exp(feature_37)")

    # gen_168 => sqrt(exp(feature_37))
    add_column("gen_168", "sqrt(abs(exp(feature_37)) + 1e-9)")

    # gen_169 => log(sqrt(exp(exp(exp(log(gen_76)))))) => 超多层
    add_column("gen_169",
               "log(abs(sqrt(abs(exp(exp(exp(log(abs(gen_76) + 1e-9))))) + 1e-9)) + 1e-9)")

    # gen_170 => 跳过

    # gen_171 => 跳过

    # gen_172 => log(exp(exp(feature_30)))
    add_column("gen_172", "log(abs(gen_72) + 1e-9)")

    # gen_173 => exp(log(sqrt(exp(exp(exp(log(gen_76))))))) => ...
    add_column("gen_173",
               "exp(log(abs(sqrt(abs(exp(exp(exp(log(abs(gen_76) + 1e-9))))) + 1e-9)) + 1e-9))")

    # gen_174 => log(gen_159)
    add_column("gen_174", "log(abs(gen_159) + 1e-9)")

    # gen_175 => exp(log(gen_159))
    add_column("gen_175", "exp(log(abs(gen_159) + 1e-9))")

    # gen_176 => sqrt(exp(log(gen_159)))
    add_column("gen_176", "sqrt(abs(gen_159) + 1e-9)")

    # gen_177 => log(gen_15)
    add_column("gen_177", "log(abs(gen_15) + 1e-9)")

    # gen_178 => exp(log(gen_15))
    add_column("gen_178", "exp(log(abs(gen_15) + 1e-9))")

    # gen_179 => sqrt(exp(log(gen_19)))
    add_column("gen_179", "sqrt(abs(gen_19) + 1e-9)")

    # gen_180 => 与 gen_174 相同 => 跳过

    # gen_181 => exp(log(gen_18))
    add_column("gen_181", "exp(log(abs(gen_18) + 1e-9))")

    # gen_182 => sqrt(exp(log(gen_15)))
    add_column("gen_182", "sqrt(abs(gen_15) + 1e-9)")

    # gen_183 => log(feature_66)
    add_column("gen_183", "log(abs(feature_66) + 1e-9)")

    # gen_184 => exp(exp(log(gen_159)))
    add_column("gen_184", "exp(exp(log(abs(gen_159) + 1e-9)))")

    # gen_185 => sqrt(feature_74)
    add_column("gen_185", "sqrt(abs(feature_74) + 1e-9)")

    # gen_186 => log(sqrt(exp(log(gen_159))))
    add_column("gen_186", "log(abs(sqrt(abs(gen_159) + 1e-9)) + 1e-9)")

    # gen_187 => exp(gen_156)
    add_column("gen_187", "exp(gen_156)")

    # gen_188 => sqrt(exp(feature_29))
    add_column("gen_188", "sqrt(abs(exp(feature_29)) + 1e-9)")

    # gen_189 => log(log(gen_159))
    add_column("gen_189", "log(abs(log(abs(gen_159) + 1e-9)) + 1e-9)")

    # gen_190 => exp(exp(log(gen_15)))
    add_column("gen_190", "exp(exp(log(abs(gen_15) + 1e-9)))")

    # gen_192 => log(sqrt(exp(log(gen_15))))
    add_column("gen_192", "log(abs(sqrt(abs(gen_15) + 1e-9)) + 1e-9)")

    # gen_193 => exp(feature_18)
    add_column("gen_193", "exp(feature_18)")

    # gen_194 => sqrt(log(feature_25))
    add_column("gen_194", "sqrt(abs(log(abs(feature_25) + 1e-9)) + 1e-9)")

    # gen_195 => log(sqrt(log(log(log(exp(log(gen_159)))))))
    add_column("gen_195",
               "log(abs(sqrt(abs(log(abs(log(abs(log(abs(exp(log(abs(gen_159) + 1e-9))) + 1e-9)) + 1e-9)) + 1e-9)) + 1e-9) ) + 1e-9)")

    return df


###########################################################################
# Dataset for iTransformer (滑窗数据集)
###########################################################################
class SlidingWindowDatasetForTransformer(Dataset):
    def __init__(self, df, feature_cols, lookback_len=17,
                 target_col='responder_6', date_col='date_id'):
        super().__init__()
        self.df = df.sort_values(['symbol_id', date_col, 'time_id']).reset_index(drop=True)
        self.feature_cols = feature_cols
        self.lookback_len = lookback_len
        self.target_col = target_col
        self.indices = []
        unique_dates = self.df[date_col].unique()
        unique_dates = np.sort(unique_dates)

        CHUNK_SIZE = 17
        for start_idx in range(0, len(unique_dates), CHUNK_SIZE):
            chunk_days = unique_dates[start_idx: start_idx + CHUNK_SIZE]
            chunk_df = self.df[self.df[date_col].isin(chunk_days)]
            if len(chunk_df) <= self.lookback_len:
                continue
            chunk_indices = chunk_df.index[-(self.lookback_len + 1):].tolist()
            if len(chunk_indices) < (self.lookback_len + 1):
                continue

            X_inds = chunk_indices[:-1]
            Y_ind = chunk_indices[-1]
            self.indices.append((X_inds, Y_ind))

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, idx):
        X_inds, Y_ind = self.indices[idx]
        x_window = self.df.loc[X_inds, self.feature_cols].values
        y_value = self.df.loc[Y_ind, self.target_col]
        return (torch.tensor(x_window, dtype=torch.float32),
                torch.tensor(y_value, dtype=torch.float32))


###########################################################################
# 一些工具函数
###########################################################################
def reduce_mem_usage(df, float16_as32=True):
    start_mem = df.memory_usage().sum() / 1024 ** 2
    print(f'[MEMORY] usage of dataframe is {start_mem:.2f} MB')
    for col in df.columns:
        if col not in df.select_dtypes(include=['object', 'category']).columns:
            col_type = df[col].dtype
            if str(col_type)[:3] == 'int':
                c_min, c_max = df[col].min(), df[col].max()
                if c_min > np.iinfo(np.int8).min and c_max < np.iinfo(np.int8).max:
                    df[col] = df[col].astype(np.int8)
                elif c_min > np.iinfo(np.int16).min and c_max < np.iinfo(np.int16).max:
                    df[col] = df[col].astype(np.int16)
                elif c_min > np.iinfo(np.int32).min and c_max < np.iinfo(np.int32).max:
                    df[col] = df[col].astype(np.int32)
                elif c_min > np.iinfo(np.int64).min and c_max < np.iinfo(np.int64).max:
                    df[col] = df[col].astype(np.int64)
            else:
                if df[col].dtype == object:
                    continue
                c_min, c_max = df[col].min(), df[col].max()
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
    print(f'[MEMORY] after optimization: {end_mem:.2f} MB')
    print(f'[MEMORY] decreased by {(start_mem - end_mem) / start_mem * 100:.1f}%')
    return df


def weighted_r2_score(y_true, y_pred, w):
    numerator = np.sum(w * (y_true - y_pred) ** 2)
    denominator = np.sum(w * (y_true - np.average(y_true, weights=w)) ** 2)
    return 1.0 - (numerator / denominator)


###########################################################################
# 遗传算法工具(完整)
###########################################################################
from deap import base, creator, tools, algorithms


def mate_and_clip(ind1, ind2, alpha=0.5):
    for i in range(len(ind1)):
        cmin = min(ind1[i], ind2[i])
        cmax = max(ind1[i], ind2[i])
        span = cmax - cmin
        lower = cmin - alpha * span
        upper = cmax + alpha * span
        ind1[i] = random.uniform(lower, upper)
        ind2[i] = random.uniform(lower, upper)
    for i in range(len(ind1)):
        ind1[i] = min(max(ind1[i], 0.0), 1.0)
        ind2[i] = min(max(ind2[i], 0.0), 1.0)
    return ind1, ind2


def mutate_and_clip(ind, eta=20.0, indpb=1.0):
    for i in range(len(ind)):
        if random.random() < indpb:
            x = ind[i]
            delta1 = x
            delta2 = 1.0 - x
            r = random.random()
            mut_pow = 1.0 / (eta + 1.0)
            if r < 0.5:
                xy = 1.0 - r
                deltaq = xy ** mut_pow - 1.0
            else:
                xy = 1.0 - r
                deltaq = 1.0 - xy ** mut_pow
            if r >= 0.5:
                x = x + deltaq * delta2
            else:
                x = x + deltaq * delta1
            x = min(max(x, 0.0), 1.0)
            ind[i] = x
    return (ind,)


def ensemble_predictions(weights, prediction_dict):
    model_names = list(prediction_dict.keys())
    assert len(weights) == len(model_names)
    final_pred = 0.0
    for w, m in zip(weights, model_names):
        final_pred += w * prediction_dict[m]
    return final_pred


def optimize_weights_genetic_algorithm(fold_predictions, y_true, w_true,
                                       population_size=50, generations=50):
    creator.create("FitnessMax", base.Fitness, weights=(1.0,))
    creator.create("Individual", list, fitness=creator.FitnessMax)

    toolbox = base.Toolbox()
    toolbox.register("attr_float", random.uniform, 0.0, 1.0)
    toolbox.register("individual", tools.initRepeat, creator.Individual,
                     toolbox.attr_float, n=len(fold_predictions))
    toolbox.register("population", tools.initRepeat, list, toolbox.individual)

    def eval_weights(individual):
        arr = np.array(individual, dtype=np.float32)
        if arr.sum() <= 1e-12:
            arr += 1e-12
        arr /= arr.sum()
        y_pred = ensemble_predictions(arr, fold_predictions)
        score = weighted_r2_score(y_true, y_pred, w_true)
        return (score,)

    toolbox.register("evaluate", eval_weights)
    toolbox.register("mate", mate_and_clip, alpha=0.5)
    toolbox.register("mutate", mutate_and_clip, eta=20.0, indpb=1.0)
    toolbox.register("select", tools.selTournament, tournsize=3)

    pop = toolbox.population(n=population_size)
    stats = tools.Statistics(lambda ind: ind.fitness.values)
    stats.register("max", np.max)
    stats.register("avg", np.mean)

    pop, logbook = algorithms.eaSimple(pop, toolbox, cxpb=0.7, mutpb=0.2,
                                       ngen=generations, stats=stats, verbose=False)

    best_ind = tools.selBest(pop, k=1)[0]
    best_arr = np.array(best_ind, dtype=np.float32)
    if best_arr.sum() <= 1e-12:
        best_arr += 1e-12
    best_arr /= best_arr.sum()
    best_score = best_ind.fitness.values[0]

    del creator.FitnessMax
    del creator.Individual
    return best_arr, best_score


def optimize_weights_genetic_algorithm_gpu(fold_predictions, y_true, w_true,
                                           population_size=50, generations=50):
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    fold_predictions_gpu = {
        m: torch.tensor(preds, dtype=torch.float32, device=device)
        for m, preds in fold_predictions.items()
    }
    y_true_gpu = torch.tensor(y_true, dtype=torch.float32, device=device)
    w_true_gpu = torch.tensor(w_true, dtype=torch.float32, device=device)

    model_list = list(fold_predictions_gpu.keys())
    num_models = len(model_list)

    creator.create("FitnessMaxGPU", base.Fitness, weights=(1.0,))
    creator.create("IndividualGPU", list, fitness=creator.FitnessMaxGPU)

    toolbox = base.Toolbox()
    toolbox.register("attr_float", random.uniform, 0.0, 1.0)
    toolbox.register("individual", tools.initRepeat, creator.IndividualGPU,
                     toolbox.attr_float, n=num_models)
    toolbox.register("population", tools.initRepeat, list, toolbox.individual)

    def eval_weights_gpu(individual):
        w_t = torch.tensor(individual, dtype=torch.float32, device=device)
        sum_w = w_t.sum()
        if sum_w < 1e-12:
            w_t = w_t + 1e-12
            sum_w = w_t.sum()
        w_t = w_t / sum_w

        fused_pred = torch.zeros_like(y_true_gpu)
        for (m_name, pred_m), w_val in zip(fold_predictions_gpu.items(), w_t):
            fused_pred += w_val * pred_m

        diff = y_true_gpu - fused_pred
        numerator = torch.sum(w_true_gpu * diff * diff)
        mean_y = torch.sum(y_true_gpu * w_true_gpu) / torch.sum(w_true_gpu)
        diff_mean = y_true_gpu - mean_y
        denominator = torch.sum(w_true_gpu * diff_mean * diff_mean)
        score = 1.0 - numerator / denominator

        return (score.item(),)

    toolbox.register("evaluate", eval_weights_gpu)
    toolbox.register("mate", mate_and_clip, alpha=0.5)
    toolbox.register("mutate", mutate_and_clip, eta=20.0, indpb=1.0)
    toolbox.register("select", tools.selTournament, tournsize=3)

    pop = toolbox.population(n=population_size)
    stats = tools.Statistics(lambda ind: ind.fitness.values)
    stats.register("max", np.max)
    stats.register("avg", np.mean)

    pop, logbook = algorithms.eaSimple(pop, toolbox, cxpb=0.7, mutpb=0.2,
                                       ngen=generations, stats=stats,
                                       verbose=False)

    best_ind = tools.selBest(pop, k=1)[0]
    best_arr = np.array(best_ind, dtype=np.float32)
    if best_arr.sum() <= 1e-12:
        best_arr += 1e-12
    best_arr /= best_arr.sum()
    best_score = best_ind.fitness.values[0]

    del creator.FitnessMaxGPU
    del creator.IndividualGPU

    return best_arr, best_score


###########################################################################
# CatBoost / XGBoost / iTransformer / Ridge 等训练函数
###########################################################################
def train_catboost_kfold(df, feature_cols, folds=5):
    """
    对 CatBoost: 指定 cat_features= 'feature_09','feature_10','feature_11' 的索引
    """
    cat_indices = [feature_cols.index(c) for c in CATEGORICAL_COLS
                   if c in feature_cols]
    models = []
    X = df[feature_cols]
    y = df[TARGET_COL]
    w = df[WEIGHT_COL]

    kf = KFold(n_splits=folds, shuffle=True, random_state=42)
    for fold_i, (tr_idx, val_idx) in enumerate(kf.split(X)):
        print(f"[CatBoost KFold] Fold {fold_i + 1}/{folds}")
        X_train, X_valid = X.iloc[tr_idx], X.iloc[val_idx]
        y_train, y_valid = y.iloc[tr_idx], y.iloc[val_idx]
        w_train, w_valid = w.iloc[tr_idx], w.iloc[val_idx]

        train_pool = cbt.Pool(X_train, label=y_train, weight=w_train,
                              cat_features=cat_indices)
        valid_pool = cbt.Pool(X_valid, label=y_valid, weight=w_valid,
                              cat_features=cat_indices)

        model = cbt.CatBoostRegressor(
            iterations=tree_iteration_num,
            learning_rate=0.01,
            depth=8,
            task_type='GPU',
            loss_function='RMSE',
            l2_leaf_reg=5,
            devices=device_id,
            verbose=False
        )
        model.fit(train_pool, eval_set=[valid_pool],
                  early_stopping_rounds=200, verbose=500)
        models.append(model)

    return models


def train_catboost_online_splitted(daily_data_dir, feature_cols, skip_dates, earliest_test_date):
    """
    同理, 在线更新时 cat_features 也一致
    """
    cat_indices = [feature_cols.index(c) for c in CATEGORICAL_COLS
                   if c in feature_cols]
    model = None
    train_day_files = _list_train_day_files(daily_data_dir, skip_dates, earliest_test_date)

    CHUNK_SIZE = 17
    for chunk_end in range(0, len(train_day_files), CHUNK_SIZE):
        files_subset = train_day_files[:(chunk_end + CHUNK_SIZE)]
        if not files_subset:
            continue

        df_chunk_list = []
        for fp in files_subset:
            df_day = pd.read_parquet(fp)
            df_chunk_list.append(df_day)
        df_chunk = pd.concat(df_chunk_list, ignore_index=True)
        del df_chunk_list

        X_chunk = df_chunk[feature_cols]
        y_chunk = df_chunk[TARGET_COL]
        w_chunk = df_chunk[WEIGHT_COL] if WEIGHT_COL in df_chunk.columns else None

        train_pool = cbt.Pool(X_chunk, label=y_chunk, weight=w_chunk,
                              cat_features=cat_indices)
        if model is None:
            model = cbt.CatBoostRegressor(
                learning_rate=0.05,
                depth=8,
                task_type='CPU',
                loss_function='RMSE',
                l2_leaf_reg=5,
                verbose=False
            )
            model.fit(train_pool)
        else:
            model.fit(train_pool, init_model=model)

        del df_chunk, X_chunk, y_chunk, w_chunk, train_pool
        gc.collect()

    return model


def train_xgb_kfold(df, feature_cols, folds=5):
    """
    XGBoost并不会把 'feature_09','feature_10','feature_11'
    当成类别列, 我们前面把它们转成 int codes, 这里就当数值即可
    """
    models = []
    X = df[feature_cols].values
    y = df[TARGET_COL].values
    w = df[WEIGHT_COL].values

    kf = KFold(n_splits=folds, shuffle=True, random_state=42)
    for fold_i, (tr_idx, val_idx) in enumerate(kf.split(X)):
        print(f"[XGBoost KFold] Fold {fold_i + 1}/{folds}")
        X_train, X_valid = X[tr_idx], X[val_idx]
        y_train, y_valid = y[tr_idx], y[val_idx]
        w_train, w_valid = w[tr_idx], w[val_idx]

        model = xgb.XGBRegressor(
            n_estimators=tree_iteration_num,
            max_depth=7,
            learning_rate=0.01,
            tree_method='hist',
            objective='reg:squarederror',
            subsample=0.8,
            colsample_bytree=0.8,
            reg_alpha=0.1,
            reg_lambda=0.05,
            gamma=1,
            n_jobs=-1,
            min_child_weight=7,
            device='cuda'
        )
        model.fit(X_train, y_train, sample_weight=w_train,
                  eval_set=[(X_valid, y_valid)],
                  sample_weight_eval_set=[w_valid],
                  early_stopping_rounds=200, verbose=100)
        models.append(model)

    return models


def train_xgb_online_splitted(daily_data_dir, feature_cols, skip_dates, earliest_test_date):
    model = None
    train_day_files = _list_train_day_files(daily_data_dir, skip_dates, earliest_test_date)

    CHUNK_SIZE = 17
    for chunk_end in range(0, len(train_day_files), CHUNK_SIZE):
        files_subset = train_day_files[:(chunk_end + CHUNK_SIZE)]
        if not files_subset:
            continue

        df_chunk_list = []
        for fp in files_subset:
            df_day = pd.read_parquet(fp)
            df_chunk_list.append(df_day)
        df_chunk = pd.concat(df_chunk_list, ignore_index=True)
        del df_chunk_list

        X_chunk = df_chunk[feature_cols].values
        y_chunk = df_chunk[TARGET_COL].values
        w_chunk = df_chunk[WEIGHT_COL].values if WEIGHT_COL in df_chunk.columns else None

        if model is None:
            model = xgb.XGBRegressor(
                n_estimators=online_tree_iteration_num,
                max_depth=7,
                learning_rate=0.05,
                tree_method='hist',
                objective='reg:squarederror',
                subsample=0.8,
                colsample_bytree=0.8,
                reg_alpha=0.1,
                reg_lambda=0.05,
                gamma=1,
                n_jobs=-1,
                min_child_weight=7,
                device='cuda'
            )
            model.fit(X_chunk, y_chunk, sample_weight=w_chunk, verbose=False)
        else:
            model.fit(X_chunk, y_chunk, sample_weight=w_chunk,
                      xgb_model=model.get_booster(), verbose=False)

        del df_chunk, X_chunk, y_chunk, w_chunk
        gc.collect()

    return model


###########################################################################
# 全新 iTransformer：内部输出每个时刻 (B, lookback_len, 1) + 因果掩码
###########################################################################
class CausalAttention(nn.Module):
    def __init__(self, dim, dim_head=32, heads=4, dropout=0.):
        super().__init__()
        self.dim_head = dim_head
        self.heads = heads
        self.scale = dim_head ** -0.5

        inner_dim = dim_head * heads
        self.norm = nn.LayerNorm(dim)

        self.to_qkv = nn.Linear(dim, inner_dim * 3, bias=False)
        self.dropout = nn.Dropout(dropout)
        self.out_proj = nn.Linear(inner_dim, dim, bias=False)

    def forward(self, x):
        """
        x: (B, N, D)
        return: (B, N, D)
        """
        b, n, d = x.shape
        h = self.heads

        x = self.norm(x)
        qkv = self.to_qkv(x)  # (B, N, 3*inner_dim)
        q, k, v = qkv.chunk(3, dim=-1)  # each: (B, N, inner_dim)

        q = q.view(b, n, h, self.dim_head).permute(0, 2, 1, 3)  # (b, h, n, d_head)
        k = k.view(b, n, h, self.dim_head).permute(0, 2, 1, 3)
        v = v.view(b, n, h, self.dim_head).permute(0, 2, 1, 3)

        # causal mask
        causal_mask = torch.triu(
            torch.ones((n, n), device=x.device) * float('-inf'),
            diagonal=1
        )
        attn_scores = torch.matmul(q, k.transpose(-2, -1)) * self.scale
        attn_scores = attn_scores + causal_mask
        attn_probs = attn_scores.softmax(dim=-1)
        attn_probs = self.dropout(attn_probs)

        out = torch.matmul(attn_probs, v)  # (b, h, n, d_head)
        out = out.permute(0, 2, 1, 3).contiguous().view(b, n, -1)
        out = self.out_proj(out)
        return out


class CausalFFN(nn.Module):
    def __init__(self, dim, ff_mult=4, dropout=0.):
        super().__init__()
        hidden = int(dim * ff_mult)
        self.net = nn.Sequential(
            nn.LayerNorm(dim),
            nn.Linear(dim, hidden),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden, dim),
            nn.Dropout(dropout),
        )

    def forward(self, x):
        return self.net(x)


class iTransformer(nn.Module):
    def __init__(
            self,
            num_variates,
            lookback_len,
            depth,
            dim,
            pred_length,
            dim_head,
            heads,
            attn_dropout,
            ff_mult,
            ff_dropout,
            num_mem_tokens,
            num_residual_streams,
            use_reversible_instance_norm,
            reversible_instance_norm_affine,
            flash_attn
    ):
        super().__init__()
        self.num_variates = num_variates
        self.lookback_len = lookback_len

        # 输入投影
        self.input_proj = nn.Linear(num_variates, dim, bias=False)

        # 堆叠多层
        self.layers = nn.ModuleList([])
        for _ in range(depth):
            attn = CausalAttention(dim, dim_head=dim_head, heads=heads, dropout=attn_dropout)
            ffn = CausalFFN(dim, ff_mult=ff_mult, dropout=ff_dropout)
            self.layers.append(nn.ModuleList([attn, ffn]))

        # 最终输出: (B, lookback_len, 1)  => 每时刻预测
        self.out_head = nn.Linear(dim, 1, bias=True)

    def forward(self, x):
        """
        x: shape=(B, lookback_len, num_variates)
        我们返回一个 dict, 其中 pred_dict[1] = (B, lookback_len, 1),
        方便外面 if isinstance(out, dict): out=out[1].
        """
        b, n, v = x.shape
        h = self.input_proj(x)  # (B, N, dim)

        for (attn, ffn) in self.layers:
            h_attn = attn(h)
            h = h + h_attn
            h_ffn = ffn(h)
            h = h + h_ffn

        out_seq = self.out_head(h)  # (B, N, 1)

        # 外层只取 out= dict[1],
        # 但这时 out_seq 里其实有 "每个时刻" 的预测
        return {1: out_seq}


###########################################################################
# iTransformer KFold & 在线学习 (保持原逻辑，外面只取最后一步)
###########################################################################
def train_itransformer_kfold(df, feature_cols, folds=4):
    from sklearn.model_selection import KFold
    models = []
    all_idx = np.arange(len(df))
    kf = KFold(n_splits=folds, shuffle=True, random_state=42)

    for fold_i, (tr_idx, val_idx) in enumerate(kf.split(all_idx)):
        print(f"[iTransformer KFold] Fold {fold_i + 1}/{folds}")
        train_sub = df.iloc[tr_idx].copy()
        valid_sub = df.iloc[val_idx].copy()

        train_sub.sort_values(DATE_COL, inplace=True)
        valid_sub.sort_values(DATE_COL, inplace=True)

        ds_valid = SlidingWindowDatasetForTransformer(
            valid_sub, feature_cols, LOOKBACK_LEN, TARGET_COL
        )
        dl_valid = DataLoader(ds_valid, batch_size=256, shuffle=False)

        base_model = iTransformer(
            num_variates=len(feature_cols),
            lookback_len=LOOKBACK_LEN,
            depth=8,
            dim=128,
            pred_length=1,
            dim_head=16,
            heads=8,
            attn_dropout=0.1,
            ff_mult=4,
            ff_dropout=0.1,
            num_mem_tokens=4,
            num_residual_streams=4,
            use_reversible_instance_norm=True,
            reversible_instance_norm_affine=True,
            flash_attn=True
        )
        model = nn.DataParallel(base_model, device_ids=device_id_list).to(DEVICE)
        optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
        criterion = nn.MSELoss()
        EPOCHS = epoch_long
        CHUNK_SIZE = 17
        unique_train_dates = train_sub[DATE_COL].unique()
        unique_train_dates = np.sort(unique_train_dates)

        for epoch in range(EPOCHS):
            model.train()
            train_losses = []
            for chunk_start in range(0, len(unique_train_dates), CHUNK_SIZE):
                chunk_days = unique_train_dates[chunk_start: chunk_start + CHUNK_SIZE]
                chunk_df = train_sub[train_sub[DATE_COL].isin(chunk_days)]
                if len(chunk_df) < (LOOKBACK_LEN + 1):
                    continue

                ds_train_chunk = SlidingWindowDatasetForTransformer(
                    chunk_df, feature_cols, LOOKBACK_LEN, TARGET_COL, DATE_COL
                )
                dl_train_chunk = DataLoader(ds_train_chunk, batch_size=256,
                                            shuffle=True, drop_last=False)

                local_losses = []
                for xb, yb in dl_train_chunk:
                    xb = xb.to(DEVICE)
                    yb = yb.to(DEVICE)
                    optimizer.zero_grad()
                    out = model(xb)
                    # out => dict, out[1] => (B, lookback_len, 1)
                    out = out[1][:, -1, 0]  # 只取最后1条
                    loss = criterion(out, yb)
                    loss.backward()
                    optimizer.step()
                    local_losses.append(loss.item())

                train_losses.extend(local_losses)
                del ds_train_chunk, dl_train_chunk, chunk_df
                gc.collect()

            model.eval()
            valid_losses = []
            with torch.no_grad():
                for xv, yv in dl_valid:
                    xv = xv.to(DEVICE)
                    yv = yv.to(DEVICE)
                    outv = model(xv)
                    outv = outv[1][:, -1, 0]
                    loss_v = criterion(outv, yv)
                    valid_losses.append(loss_v.item())

            print(f"   Fold={fold_i + 1}, Epoch={epoch + 1}, "
                  f"TrainMSE={np.mean(train_losses):.6f}, "
                  f"ValidMSE={np.mean(valid_losses):.6f}")

        models.append(model)
        del ds_valid, dl_valid, train_sub, valid_sub
        gc.collect()

    return models


def train_itransformer_online_splitted(daily_data_dir, feature_cols, skip_dates, earliest_test_date):
    """
    改动要点：
    1) 不再只取 out[:, -1, 0]；而是 out.view(-1) ，表示拿到 (batch_size * lookback_len) 条预测
    2) 把 yb 扩展到同样尺寸(只是将最后一个label复制给各时刻)
    3) 同步 reshape => yb = yb.unsqueeze(1).expand(-1, LOOKBACK_LEN).reshape(-1)
    """

    base_model = iTransformer(
        num_variates=len(feature_cols),
        lookback_len=LOOKBACK_LEN,
        depth=8,
        dim=128,
        pred_length=1,
        dim_head=16,
        heads=8,
        attn_dropout=0.1,
        ff_mult=4,
        ff_dropout=0.1,
        num_mem_tokens=4,
        num_residual_streams=4,
        use_reversible_instance_norm=True,
        reversible_instance_norm_affine=True,
        flash_attn=True
    )
    model = nn.DataParallel(base_model, device_ids=device_id_list).to(DEVICE)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    criterion = nn.MSELoss()

    train_day_files = _list_train_day_files(daily_data_dir, skip_dates, earliest_test_date)
    WINDOW_SIZE = 17
    EPOCHS_THIS_CHUNK = epoch_short

    for current_end_idx in range(len(train_day_files)):
        chunk_start_idx = max(0, current_end_idx - WINDOW_SIZE + 1)
        chunk_files = train_day_files[chunk_start_idx: current_end_idx + 1]
        if not chunk_files:
            continue

        df_chunk_list = []
        for cf in chunk_files:
            df_day = pd.read_parquet(cf)
            df_chunk_list.append(df_day)
        df_chunk = pd.concat(df_chunk_list, ignore_index=True)
        del df_chunk_list

        ds_chunk = SlidingWindowDatasetForTransformer(df_chunk, feature_cols, LOOKBACK_LEN, TARGET_COL, DATE_COL)
        dl_chunk = DataLoader(ds_chunk, batch_size=256, shuffle=True, drop_last=False)

        print(f"[iTransformer Online] dayIndex {current_end_idx}, total chunk files={len(chunk_files)}")
        for ep in range(EPOCHS_THIS_CHUNK):
            model.train()
            losses = []
            for xb, yb in dl_chunk:
                xb = xb.to(DEVICE)  # shape=(batch_size, lookback_len, num_features)
                yb = yb.to(DEVICE)  # shape=(batch_size,)

                optimizer.zero_grad()
                out = model(xb)
                if isinstance(out, dict):
                    # out: (batch_size, lookback_len, 1)
                    out = out[1]

                # —— 原来只取最后一个时间步 out[:, -1, 0] ——
                # —— 现在改为整个序列每个时刻都输出 ——

                # (1) 拉平预测: => (batch_size * lookback_len,)
                out = out.view(-1)

                # (2) 扩展 yb 让它匹配 out 的长度
                #     由于当前Dataset只有1条label对应滑窗最后一行，这里只是将那1条label“广播”
                #     到所有时间步(同一个值多份拷贝)，以便维度匹配
                yb_expanded = yb.unsqueeze(1).expand(-1, LOOKBACK_LEN).reshape(-1)

                loss = criterion(out, yb_expanded)
                loss.backward()
                optimizer.step()
                losses.append(loss.item())

            print(f"   => ep={ep + 1}, meanLoss={np.mean(losses):.6f}")

        del df_chunk, ds_chunk, dl_chunk
        gc.collect()

    return model


def train_ridge_kfold(df, feature_cols, folds=5):
    models = []
    X = df[feature_cols].values
    y = df[TARGET_COL].values
    w = df[WEIGHT_COL].values

    kf = KFold(n_splits=folds, shuffle=True, random_state=42)
    for fold_i, (tr_idx, val_idx) in enumerate(kf.split(X)):
        print(f"[Ridge KFold] Fold {fold_i + 1}/{folds}")
        X_train, X_valid = X[tr_idx], X[val_idx]
        y_train, y_valid = y[tr_idx], y[val_idx]
        w_train, w_valid = w[tr_idx], w[val_idx]

        model = Ridge(alpha=0.1)
        model.fit(X_train, y_train, sample_weight=w_train)
        models.append(model)

    return models


def predict_itransformer_kfold(models, df_test, raw_feature_cols):
    """
    => 分块 + 重叠的方式，让 iTransformer 在推理时对 df_test 每行都输出预测，不爆内存
    """
    chunk_size = 10000
    overlap = LOOKBACK_LEN  # 17
    # 1) 排序 + 记录原index
    df_test_sorted = df_test.sort_values(['symbol_id', 'date_id', 'time_id']).reset_index()
    original_idx = df_test_sorted['index'].values
    X_all = df_test_sorted[raw_feature_cols].values

    # 2) 分块处理 + 重叠
    N = len(X_all)
    preds_all = np.zeros(N, dtype=np.float32)

    # 累积输出位置
    start_pos = 0
    # chunk: [i, i+chunk_size)
    i = 0
    while i < N:
        # chunk 起点
        chunk_start = i
        # chunk 终点
        chunk_end = min(i + chunk_size, N)

        # 取 [chunk_start - overlap, chunk_end)
        # 但是不能小于0
        context_start = max(0, chunk_start - overlap)
        # 做一个局部X
        X_chunk = X_all[context_start:chunk_end]

        # 构建 (1, seq_len, num_variates)
        seq_len = X_chunk.shape[0]
        x_torch = torch.tensor(X_chunk, dtype=torch.float32, device=DEVICE).unsqueeze(0)

        # 推理
        # out: (1, seq_len, 1)
        out = np.zeros(seq_len, dtype=np.float32)
        with torch.no_grad():
            # 多折模型取平均
            fold_preds_sum = np.zeros(seq_len, dtype=np.float32)
            for m in models:
                m.eval()
                out_m = m(x_torch)  # dict => {1: (1, seq_len, 1)}
                out_m = out_m[1]  # shape=(1, seq_len, 1)
                out_m = out_m.squeeze().cpu().numpy()  # (seq_len,)
                fold_preds_sum += out_m
            # 取平均
            out = fold_preds_sum / len(models)

        # 跳过 overlap
        if context_start == 0:
            # 第一个chunk无需跳过
            valid_part = out
        else:
            skip_len = chunk_start - context_start
            valid_part = out[skip_len:]

        # 放到 preds_all
        valid_len = len(valid_part)
        preds_all[chunk_start: chunk_start + valid_len] = valid_part

        i += (chunk_size - overlap)  # 前移 (chunk_size - overlap)

    # 恢复到原先排序
    # preds_all 对应df_test_sorted行顺序, 需要放回 original_idx
    final_preds = np.zeros(N, dtype=np.float32)
    final_preds[df_test_sorted.index] = preds_all

    return final_preds


def predict_itransformer_online(model, df_test, raw_feature_cols):
    """
    => 分块 + 重叠 (2.1)
    同理，但只有单个模型(online)，不做多折平均
    """
    chunk_size = 10000
    overlap = LOOKBACK_LEN  # 17
    df_test_sorted = df_test.sort_values(['symbol_id', 'date_id', 'time_id']).reset_index()
    original_idx = df_test_sorted['index'].values
    X_all = df_test_sorted[raw_feature_cols].values

    N = len(X_all)
    preds_all = np.zeros(N, dtype=np.float32)

    i = 0
    while i < N:
        chunk_start = i
        chunk_end = min(i + chunk_size, N)
        context_start = max(0, chunk_start - overlap)
        X_chunk = X_all[context_start:chunk_end]
        seq_len = X_chunk.shape[0]

        x_torch = torch.tensor(X_chunk, dtype=torch.float32, device=DEVICE).unsqueeze(0)
        with torch.no_grad():
            model.eval()
            out_m = model(x_torch)  # {1: (1, seq_len, 1)}
            out_m = out_m[1].squeeze().cpu().numpy()  # shape=(seq_len,)

        if context_start == 0:
            valid_part = out_m
        else:
            skip_len = chunk_start - context_start
            valid_part = out_m[skip_len:]
        valid_len = len(valid_part)
        preds_all[chunk_start: chunk_start + valid_len] = valid_part

        i += (chunk_size - overlap)

    final_preds = np.zeros(N, dtype=np.float32)
    final_preds[df_test_sorted.index] = preds_all
    return final_preds


###########################################################################
# 只列出 [skip_date, earliest_test_date) 范围内的 day_{xxx}.parquet
###########################################################################
def _list_train_day_files(daily_data_dir, skip_dates, earliest_test_date):
    all_day_files = glob(os.path.join(daily_data_dir, "day_*.parquet"))
    valid_files = []
    for fp in all_day_files:
        bn = os.path.basename(fp)
        d_str = bn.replace("day_", "").replace(".parquet", "")
        d_int = int(d_str)
        if (d_int >= skip_dates) and (d_int < earliest_test_date):
            valid_files.append(fp)
    valid_files.sort(key=lambda fp: int(os.path.basename(fp)
                                        .replace("day_", "").replace(".parquet", "")))
    return valid_files


###########################################################################
# 主流程
###########################################################################
def main():
    if not TRAINING:
        print("TRAINING = False, skipping training")
        return

    # 1) 读入大数据
    if not os.path.exists(TRAIN_PATH) or os.path.getsize(TRAIN_PATH) == 0:
        print(f"[ERROR] {TRAIN_PATH} not found or empty.")
        return

    df = pd.read_parquet(TRAIN_PATH)
    df = df.fillna(3.0)
    df = reduce_mem_usage(df, float16_as32=False)
    df.sort_values(['symbol_id', 'date_id', 'time_id'], inplace=True, ignore_index=True)

    # 做简单lag
    lag_cols = [f"responder_{i}" for i in range(9)]
    last_records = df.groupby(['symbol_id', 'date_id'])[lag_cols].last().reset_index()
    last_records['date_id'] += 1
    lags = pd.merge(df, last_records, on=['symbol_id', 'date_id'], how='left', suffixes=('', '_lag_1'))
    new_lag_cols = [f"{col}_lag_1" for col in lag_cols]
    lags = lags.dropna(subset=new_lag_cols).reset_index(drop=True)

    # skip
    lags = lags[lags['date_id'] >= SKIP_DATES].copy()
    lags.sort_values('date_id', inplace=True)
    lags.reset_index(drop=True, inplace=True)

    # 不做任何频率/独热/目标编码; CatBoost会自动编码, XGB等则把它当数值(整数)
    for c in CATEGORICAL_COLS:
        if c in lags.columns:
            lags[c] = lags[c].astype('category').cat.as_ordered()
            lags[c] = lags[c].cat.codes.astype(np.int32)

    all_cols = lags.columns.tolist()
    feature_cols = [col for col in all_cols if col not in EXCLUDE_COLS and col in lags.columns]

    # ===== 划分训练/验证/测试 =====
    all_dates = lags['date_id'].unique()
    all_dates.sort()
    test_dates = all_dates[-NUM_TEST_DATES:]
    valid_dates = all_dates[-(NUM_TEST_DATES + NUM_VALID_DATES): -NUM_TEST_DATES]
    train_dates = all_dates[: -(NUM_TEST_DATES + NUM_VALID_DATES)]
    earliest_valid_date = valid_dates[0]
    earliest_test_date = test_dates[0]

    test_df = lags[lags['date_id'].isin(test_dates)].reset_index(drop=True)
    valid_df = lags[lags['date_id'].isin(valid_dates)].reset_index(drop=True)
    train_online_df = lags[lags['date_id'].isin(train_dates)].reset_index(drop=True)

    # =========== (A) K_FOLD 训练(示例) ===============
    cat_kfold_models = train_catboost_kfold(train_online_df, feature_cols, folds=4)
    for i, catm in enumerate(cat_kfold_models):
        catm.save_model(os.path.join(MODEL_PATH, f"catboost_kfold_fold{i}.cbm"))

    xgb_kfold_models = train_xgb_kfold(train_online_df, feature_cols, folds=4)
    for i, xgm in enumerate(xgb_kfold_models):
        xgm.save_model(os.path.join(MODEL_PATH, f"xgb_kfold_fold{i}.json"))

    itr_kfold_models = train_itransformer_kfold(train_online_df, feature_cols, folds=4)
    for i, itm in enumerate(itr_kfold_models):
        torch.save(itm.state_dict(), os.path.join(MODEL_PATH, f"itransformer_kfold_fold{i}.pth"))

    ridge_kfold_models = train_ridge_kfold(train_online_df, feature_cols, folds=4)
    for i, rm in enumerate(ridge_kfold_models):
        joblib.dump(rm, os.path.join(MODEL_PATH, f"ridge_kfold_fold{i}.pkl"))

    del df
    gc.collect()

    # =========== (B) 在线学习(仅用 [SKIP_DATES, earliest_valid_date) ) ===========
    cat_online_model_final = train_catboost_online_splitted(
        DAY_SPLIT_DIR, feature_cols, SKIP_DATES, earliest_valid_date
    )
    cat_online_model_final.save_model(os.path.join(MODEL_PATH, "catboost_online_final.cbm"))

    xgb_online_model_final = train_xgb_online_splitted(
        DAY_SPLIT_DIR, feature_cols, SKIP_DATES, earliest_valid_date
    )
    xgb_online_model_final.save_model(os.path.join(MODEL_PATH, "xgb_online_final.json"))

    itr_online_model_final = train_itransformer_online_splitted(
        DAY_SPLIT_DIR, feature_cols, SKIP_DATES, earliest_valid_date
    )
    torch.save(itr_online_model_final.state_dict(), os.path.join(MODEL_PATH, "itransformer_online_final.pth"))

    gc.collect()

    ########################################################################
    # 定义预测函数 (CatBoost / XGB / Ridge 不变)
    ########################################################################
    def predict_catboost_kfold(models, df_test, raw_feature_cols):
        cat_indices = [
            raw_feature_cols.index(c)
            for c in CATEGORICAL_COLS
            if c in raw_feature_cols
        ]
        test_pool = cbt.Pool(
            data=df_test[raw_feature_cols],
            cat_features=cat_indices
        )
        preds_list = []
        for m in models:
            preds_list.append(m.predict(test_pool))
        return np.mean(preds_list, axis=0)

    def predict_catboost_online(model, df_test, raw_feature_cols):
        cat_indices = [
            raw_feature_cols.index(c)
            for c in CATEGORICAL_COLS
            if c in raw_feature_cols
        ]
        test_pool = cbt.Pool(
            data=df_test[raw_feature_cols],
            cat_features=cat_indices
        )
        return model.predict(test_pool)

    def predict_xgb_kfold(models, df_test, raw_feature_cols):
        X_test = df_test[raw_feature_cols].values
        preds_list = []
        for m in models:
            preds_list.append(m.predict(X_test))
        return np.mean(preds_list, axis=0)

    def predict_xgb_online(model, df_test, raw_feature_cols):
        X_test = df_test[raw_feature_cols].values
        return model.predict(X_test)

    def predict_ridge_kfold(models, df_test, raw_feature_cols):
        X_test = df_test[raw_feature_cols].values
        preds_list = []
        for m in models:
            preds_list.append(m.predict(X_test))
        return np.mean(preds_list, axis=0)

    ########################################################################
    # 分块 + 因果掩码 => iTransformer KFold
    ########################################################################
    def predict_itransformer_kfold(models, df_test, raw_feature_cols):
        """
        => 2.1 分块 + overlap
        => 多折模型取平均
        """
        chunk_size = 2000000
        overlap = LOOKBACK_LEN  # 17

        # 1) 排序 & 记录原index
        df_sorted = df_test.sort_values(['symbol_id', 'date_id', 'time_id']).reset_index()
        original_idx = df_sorted['index'].values
        X_ = df_sorted[raw_feature_cols].values

        N = len(X_)
        preds_all = np.zeros(N, dtype=np.float32)

        i = 0
        while i < N:
            chunk_start = i
            chunk_end = min(i + chunk_size, N)
            context_start = max(0, chunk_start - overlap)

            X_chunk = X_[context_start: chunk_end]
            seq_len = X_chunk.shape[0]

            x_t = torch.tensor(X_chunk, dtype=torch.float32, device=DEVICE).unsqueeze(0)
            # => shape=(1, seq_len, #features)

            # 多折模型预测求平均
            fold_sum = np.zeros(seq_len, dtype=np.float32)
            with torch.no_grad():
                for m in models:
                    m.eval()
                    out_dict = m(x_t)  # {1: (1, seq_len, 1)}
                    out_arr = out_dict[1].squeeze().cpu().numpy()  # (seq_len,)
                    fold_sum += out_arr
            fold_pred = fold_sum / len(models)

            # 跳过前 overlap
            if context_start == 0:
                # 第1个chunk，无需跳过
                valid_part = fold_pred
            else:
                skip_len = chunk_start - context_start
                valid_part = fold_pred[skip_len:]

            valid_len = len(valid_part)
            preds_all[chunk_start: chunk_start + valid_len] = valid_part

            i += (chunk_size - overlap)

        # 恢复顺序
        final_preds = np.zeros(N, dtype=np.float32)
        final_preds[df_sorted.index] = preds_all
        return final_preds

    ########################################################################
    # 分块 + 因果掩码 => iTransformer Online
    ########################################################################
    def predict_itransformer_online(model, df_test, raw_feature_cols):
        """
        => 2.1 分块 + overlap
        => 单模型 (online)
        """
        chunk_size = 2000000
        overlap = LOOKBACK_LEN  # 17

        df_sorted = df_test.sort_values(['symbol_id', 'date_id', 'time_id']).reset_index()
        original_idx = df_sorted['index'].values
        X_ = df_sorted[raw_feature_cols].values

        N = len(X_)
        preds_all = np.zeros(N, dtype=np.float32)

        i = 0
        while i < N:
            chunk_start = i
            chunk_end = min(i + chunk_size, N)
            context_start = max(0, chunk_start - overlap)

            X_chunk = X_[context_start: chunk_end]
            seq_len = X_chunk.shape[0]

            x_t = torch.tensor(X_chunk, dtype=torch.float32, device=DEVICE).unsqueeze(0)
            with torch.no_grad():
                model.eval()
                out_dict = model(x_t)  # {1: (1, seq_len, 1)}
                out_arr = out_dict[1].squeeze().cpu().numpy()

            if context_start == 0:
                valid_part = out_arr
            else:
                skip_len = chunk_start - context_start
                valid_part = out_arr[skip_len:]

            valid_len = len(valid_part)
            preds_all[chunk_start: chunk_start + valid_len] = valid_part

            i += (chunk_size - overlap)

        final_preds = np.zeros(N, dtype=np.float32)
        final_preds[df_sorted.index] = preds_all
        return final_preds

    ############################################################################
    # 做测试集预测
    ############################################################################
    # catboost kfold
    cat_kfold_models = []
    for i in range(4):
        cmodel = cbt.CatBoostRegressor()
        cmodel.load_model(os.path.join(MODEL_PATH, f"catboost_kfold_fold{i}.cbm"))
        cat_kfold_models.append(cmodel)
    catboost_kfold_pred = predict_catboost_kfold(cat_kfold_models, test_df, feature_cols)

    # catboost online
    cat_online_model_final_loaded = cbt.CatBoostRegressor()
    cat_online_model_final_loaded.load_model(os.path.join(MODEL_PATH, "catboost_online_final.cbm"))
    catboost_online_pred = predict_catboost_online(cat_online_model_final_loaded, test_df, feature_cols)

    # xgb kfold
    xgb_kfold_models = []
    for i in range(4):
        xg = xgb.XGBRegressor()
        xg.load_model(os.path.join(MODEL_PATH, f"xgb_kfold_fold{i}.json"))
        xgb_kfold_models.append(xg)
    xgb_kfold_pred = predict_xgb_kfold(xgb_kfold_models, test_df, feature_cols)

    # xgb online
    xgb_online_model_final_loaded = xgb.XGBRegressor()
    xgb_online_model_final_loaded.load_model(os.path.join(MODEL_PATH, "xgb_online_final.json"))
    xgb_online_pred = predict_xgb_online(xgb_online_model_final_loaded, test_df, feature_cols)

    # iTransformer kfold
    itr_kfold_models_loaded = []
    for i in range(4):
        base_model = iTransformer(
            num_variates=len(feature_cols),
            lookback_len=LOOKBACK_LEN,
            depth=8,
            dim=128,
            pred_length=1,
            dim_head=16,
            heads=8,
            attn_dropout=0.1,
            ff_mult=4,
            ff_dropout=0.1,
            num_mem_tokens=4,
            num_residual_streams=4,
            use_reversible_instance_norm=True,
            reversible_instance_norm_affine=True,
            flash_attn=True
        )
        m_i = nn.DataParallel(base_model, device_ids=device_id_list).to(DEVICE)
        m_i.load_state_dict(torch.load(os.path.join(MODEL_PATH, f"itransformer_kfold_fold{i}.pth")))
        itr_kfold_models_loaded.append(m_i)
    itr_kfold_pred = predict_itransformer_kfold(itr_kfold_models_loaded, test_df, feature_cols)

    # iTransformer online
    base_online = iTransformer(
        num_variates=len(feature_cols),
        lookback_len=LOOKBACK_LEN,
        depth=8,
        dim=128,
        pred_length=1,
        dim_head=16,
        heads=8,
        attn_dropout=0.1,
        ff_mult=4,
        ff_dropout=0.1,
        num_mem_tokens=4,
        num_residual_streams=4,
        use_reversible_instance_norm=True,
        reversible_instance_norm_affine=True,
        flash_attn=True
    )
    itr_online_model_final_loaded = nn.DataParallel(base_online, device_ids=device_id_list).to(DEVICE)
    itr_online_model_final_loaded.load_state_dict(torch.load(os.path.join(MODEL_PATH, "itransformer_online_final.pth")))
    itr_online_pred = predict_itransformer_online(itr_online_model_final_loaded, test_df, feature_cols)

    # Ridge kfold
    ridge_kfold_models = []
    for i in range(4):
        rmodel = joblib.load(os.path.join(MODEL_PATH, f"ridge_kfold_fold{i}.pkl"))
        ridge_kfold_models.append(rmodel)
    ridge_kfold_pred = predict_ridge_kfold(ridge_kfold_models, test_df, feature_cols)

    # 计算加权R^2
    test_y = test_df[TARGET_COL].values
    if WEIGHT_COL in test_df.columns:
        test_w = test_df[WEIGHT_COL].values
    else:
        test_w = np.ones(len(test_df), dtype=np.float32)

    single_model_results = {
        "catboost_kfold": weighted_r2_score(test_y, catboost_kfold_pred, test_w),
        "catboost_online": weighted_r2_score(test_y, catboost_online_pred, test_w),
        "xgb_kfold": weighted_r2_score(test_y, xgb_kfold_pred, test_w),
        "xgb_online": weighted_r2_score(test_y, xgb_online_pred, test_w),
        "itransformer_kfold": weighted_r2_score(test_y, itr_kfold_pred, test_w),
        "itransformer_online": weighted_r2_score(test_y, itr_online_pred, test_w),
        "ridge_kfold": weighted_r2_score(test_y, ridge_kfold_pred, test_w),
    }
    print("\n====== 单模型（Test Set）加权R^2结果 ======")
    for k, v in single_model_results.items():
        print(f"{k}: {v:.6f}")

    # 做模型融合/遗传算法搜索(可选)
    all_model_preds = {
        "catboost_kfold": catboost_kfold_pred,
        "catboost_online": catboost_online_pred,
        "xgb_kfold": xgb_kfold_pred,
        "xgb_online": xgb_online_pred,
        "itransformer_kfold": itr_kfold_pred,
        "itransformer_online": itr_online_pred,
        "ridge_kfold": ridge_kfold_pred
    }

    from itertools import product
    cat_opts = ["none", "catboost_kfold", "catboost_online"]
    xgb_opts = ["none", "xgb_kfold", "xgb_online"]
    itr_opts = ["none", "itransformer_kfold", "itransformer_online"]
    rdg_opts = ["none", "ridge_kfold"]

    combo_results = []
    for c_opt, x_opt, i_opt, r_opt in product(cat_opts, xgb_opts, itr_opts, rdg_opts):
        if c_opt == "none" and x_opt == "none" and i_opt == "none" and r_opt == "none":
            continue
        chosen_models = []
        if c_opt != "none": chosen_models.append(c_opt)
        if x_opt != "none": chosen_models.append(x_opt)
        if i_opt != "none": chosen_models.append(i_opt)
        if r_opt != "none": chosen_models.append(r_opt)

        preds_dict = {m: all_model_preds[m] for m in chosen_models}
        best_w, best_s = optimize_weights_genetic_algorithm_gpu(
            preds_dict, test_y, test_w,
            population_size=30,
            generations=30
        )
        combo_results.append((chosen_models, best_s, best_w))

    combo_results.sort(key=lambda x: x[1], reverse=True)
    print("\n====== 不同模型组合(权重由遗传算法确定)在Test Set上的加权R2，从高到低排序 ======")
    for rank_i, (ms_used, sc_used, wt_used) in enumerate(combo_results, start=1):
        wt_str = ", ".join([f"{m}={round(w, 4)}" for m, w in zip(ms_used, wt_used)])
        line_str = f"{rank_i}. Models={ms_used}, Weighted_R2={sc_used:.6f}, Weights=[{wt_str}]"

        print(line_str)

        with open(ROOT_DIR + "best_comb.txt", "a", encoding="utf-8") as f:
            f.write(line_str + "\n")

    print("\n===== 训练与组合完毕！=====")


if __name__ == "__main__":
    main()
