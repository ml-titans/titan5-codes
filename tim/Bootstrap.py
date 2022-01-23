# -*- coding: utf-8 -*-
# ---
# jupyter:
#   jupytext:
#     text_representation:
#       extension: .py
#       format_name: light
#       format_version: '1.5'
#       jupytext_version: 1.13.6
#   kernelspec:
#     display_name: Python 3 (ipykernel)
#     language: python
#     name: python3
# ---

# # Bootstrap

# ## 準備

# +
import numpy as np
import pandas as pd
import graphviz
import lingam
from lingam.utils import print_causal_directions, print_dagc, make_dot
import warnings


# warningを非表示
warnings.filterwarnings('ignore')

# numpyは小数第3位まで表示
np.set_printoptions(precision=3, suppress=True)

# 乱数を設定
np.random.seed(0)
# -

# ## テストデータ

# +
# 各変数ごとにデータ生成
x3 = np.random.uniform(size=1000)
x0 = 3.0*x3 + np.random.uniform(size=1000)
x2 = 6.0*x3 + np.random.uniform(size=1000)
x1 = 3.0*x0 + 2.0*x2 + np.random.uniform(size=1000)
x5 = 4.0*x0 + np.random.uniform(size=1000)
x4 = 8.0*x0 - 1.0*x2 + np.random.uniform(size=1000)

# DataFrameとして格納
X = pd.DataFrame(np.array([x0, x1, x2, x3, x4, x5]).T ,columns=['x0', 'x1', 'x2', 'x3', 'x4', 'x5'])
# -

# ## ブートストラップ法

# +
# DirectLiNGAMのオブジェクト
model = lingam.DirectLiNGAM()

# ブートストラップ法（サンプリング数100）で学習
result = model.bootstrap(X, n_sampling=100)
# -

# ## 因果方向

# ブートストラップ法の結果に基づき因果の矢印を確度の高い順で集計
# n_directions: 集計する因果の矢印の数（ランキング順）
# min_causal_effect: 集計する係数（因果効果）の最小値
# split_by_causal_effect_sign: 係数（因果効果）の符号を区別するかどうか
cdc = result.get_causal_direction_counts(n_directions=8, min_causal_effect=0.01, split_by_causal_effect_sign=True)

# 集計結果の表示
# 第2引数（=100）はブートストラップ法のサンプリング数を入力
print_causal_directions(cdc, 100)

# ## 有向非巡回グラフ

# ブートストラップ法の結果に基づき因果構造を確度の高い順で集計
# n_dags: 集計する因果構造の数（ランキング順）
# min_causal_effect: 考慮する係数（因果効果）の最小値
# split_by_causal_effect_sign: 係数（因果効果）の符号を区別するかどうか
dagc = result.get_directed_acyclic_graph_counts(n_dags=3, min_causal_effect=0.01, split_by_causal_effect_sign=True)

# 集計結果の表示
# 第2引数（=100）はブートストラップサンプル数を入力
print_dagc(dagc, 100)

# ## 出現確率

# +
# ブートストラップ法の結果に基づき各因果の矢印の出現確率を集計
# min_causal_effect: 考慮する係数（因果効果）の最小値
prob = result.get_probabilities(min_causal_effect=0.01)

# 集計結果の表示
print(prob)
# -

# ## （総合）因果効果

# +
# ブートストラップ法の結果に基づき（総合）因果効果を計算
# min_causal_effect: 考慮する係数（因果効果）の最小値
causal_effects = result.get_total_causal_effects(min_causal_effect=0.01)

# DataFrameに結果をまとめる
df = pd.DataFrame(causal_effects)
labels = [f'x{i}' for i in range(X.shape[1])]
df['from'] = df['from'].apply(lambda x : labels[x])
df['to'] = df['to'].apply(lambda x : labels[x])
# -

# （総合）因果効果の大きい順にトップ5を表示
print(df.sort_values('effect', ascending=False).head())

# 存在確率の小さい順にトップ5を表示
print(df.sort_values('probability', ascending=True).head())

# x1に向かう（総合）因果効果を表示
print(df[df['to']=='x1'].head())

# ## 経路の存在確率と（総合）因果効果

# +
# ブートストラップ法の結果に基づき経路の存在確率と（総合）因果効果を計算
from_index = 3 # 原因となる変数のインデックス（x3）
to_index = 1 # 結果となる変数のインデックス（x1）

# 存在確率の大きい順にトップ5を表示
print(pd.DataFrame(result.get_paths(from_index, to_index)).head())
# -


