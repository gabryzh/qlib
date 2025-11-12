# coding=utf-8
# Copyright 2020 The Google Research Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

# Lint as: python3
"""时序融合 Transformer 模型。

包含完整的 TFT 架构和相关组件。定义了使用简单的 Pandas 数据帧输入进行训练、评估和预测的函数。
"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import gc
import json
import os
import shutil

import data_formatters.base
import libs.utils as utils
import numpy as np
import pandas as pd
import tensorflow as tf

# 层定义。
concat = tf.keras.backend.concatenate
stack = tf.keras.backend.stack
K = tf.keras.backend
Add = tf.keras.layers.Add
LayerNorm = tf.keras.layers.LayerNormalization
Dense = tf.keras.layers.Dense
Multiply = tf.keras.layers.Multiply
Dropout = tf.keras.layers.Dropout
Activation = tf.keras.layers.Activation
Lambda = tf.keras.layers.Lambda

# 默认输入类型。
InputTypes = data_formatters.base.InputTypes


# 层实用函数。
def linear_layer(size, activation=None, use_time_distributed=False, use_bias=True):
    """返回简单的 Keras 线性层。

    参数:
      size: 输出大小
      activation: 如果需要，应用的激活函数
      use_time_distributed: 是否跨时间应用层
      use_bias: 层中是否应包含偏置

    返回:
        一个 Keras 线性层。
    """
    linear = tf.keras.layers.Dense(size, activation=activation, use_bias=use_bias)
    if use_time_distributed:
        linear = tf.keras.layers.TimeDistributed(linear)
    return linear


def apply_mlp(
    inputs, hidden_size, output_size, output_activation=None, hidden_activation="tanh", use_time_distributed=False
):
    """将简单的前馈网络应用于输入。

    参数:
      inputs: MLP 输入
      hidden_size: 隐藏状态大小
      output_size: MLP 的输出大小
      output_activation: 在输出上应用的激活函数
      hidden_activation: 在输入上应用的激活函数
      use_time_distributed: 是否跨时间应用

    返回:
      MLP 输出的张量。
    """
    if use_time_distributed:
        hidden = tf.keras.layers.TimeDistributed(tf.keras.layers.Dense(hidden_size, activation=hidden_activation))(
            inputs
        )
        return tf.keras.layers.TimeDistributed(tf.keras.layers.Dense(output_size, activation=output_activation))(hidden)
    else:
        hidden = tf.keras.layers.Dense(hidden_size, activation=hidden_activation)(inputs)
        return tf.keras.layers.Dense(output_size, activation=output_activation)(hidden)


def apply_gating_layer(x, hidden_layer_size, dropout_rate=None, use_time_distributed=True, activation=None):
    """将门控线性单元 (GLU) 应用于输入。

    参数:
      x: 门控层的输入
      hidden_layer_size: GLU 的维度
      dropout_rate: 如果有，应用的 dropout 率
      use_time_distributed: 是否跨时间应用
      activation: 如果需要，应用于线性特征变换的激活函数

    返回:
      张量元组: (GLU 输出, 门)
    """

    if dropout_rate is not None:
        x = tf.keras.layers.Dropout(dropout_rate)(x)

    if use_time_distributed:
        activation_layer = tf.keras.layers.TimeDistributed(
            tf.keras.layers.Dense(hidden_layer_size, activation=activation)
        )(x)
        gated_layer = tf.keras.layers.TimeDistributed(tf.keras.layers.Dense(hidden_layer_size, activation="sigmoid"))(x)
    else:
        activation_layer = tf.keras.layers.Dense(hidden_layer_size, activation=activation)(x)
        gated_layer = tf.keras.layers.Dense(hidden_layer_size, activation="sigmoid")(x)

    return tf.keras.layers.Multiply()([activation_layer, gated_layer]), gated_layer


def add_and_norm(x_list):
    """应用跳跃连接，然后进行层归一化。

    参数:
      x_list: 用于跳跃连接求和的输入列表

    返回:
      层的张量输出。
    """
    tmp = Add()(x_list)
    tmp = LayerNorm()(tmp)
    return tmp


def gated_residual_network(
    x,
    hidden_layer_size,
    output_size=None,
    dropout_rate=None,
    use_time_distributed=True,
    additional_context=None,
    return_gate=False,
):
    """应用论文中定义的门控残差网络 (GRN)。

    参数:
      x: 网络输入
      hidden_layer_size: 内部状态大小
      output_size: 输出层的大小
      dropout_rate: 如果应用了 dropout，则为 dropout 率
      use_time_distributed: 是否跨时间维度应用网络
      additional_context: 如果相关，要使用的附加上下文向量
      return_gate: 是否返回 GLU 门以用于诊断目的

    返回:
      张量元组: (GRN 输出, GLU 门)
    """

    # 设置跳跃连接
    if output_size is None:
        output_size = hidden_layer_size
        skip = x
    else:
        linear = Dense(output_size)
        if use_time_distributed:
            linear = tf.keras.layers.TimeDistributed(linear)
        skip = linear(x)

    # 应用前馈网络
    hidden = linear_layer(hidden_layer_size, activation=None, use_time_distributed=use_time_distributed)(x)
    if additional_context is not None:
        hidden = hidden + linear_layer(
            hidden_layer_size, activation=None, use_time_distributed=use_time_distributed, use_bias=False
        )(additional_context)
    hidden = tf.keras.layers.Activation("elu")(hidden)
    hidden = linear_layer(hidden_layer_size, activation=None, use_time_distributed=use_time_distributed)(hidden)

    gating_layer, gate = apply_gating_layer(
        hidden, output_size, dropout_rate=dropout_rate, use_time_distributed=use_time_distributed, activation=None
    )

    if return_gate:
        return add_and_norm([skip, gating_layer]), gate
    else:
        return add_and_norm([skip, gating_layer])


# 注意力组件。
def get_decoder_mask(self_attn_inputs):
    """返回要应用于自注意力层的因果掩码。

    参数:
      self_attn_inputs: 自注意力层的输入，用于确定掩码形状
    """
    len_s = tf.shape(self_attn_inputs)[1]
    bs = tf.shape(self_attn_inputs)[:1]
    mask = K.cumsum(tf.eye(len_s, batch_shape=bs), 1)
    return mask


class ScaledDotProductAttention:
    """定义缩放点积注意力层。

    属性:
      dropout: 要使用的 dropout 率
      activation: 缩放点积注意力的归一化函数 (例如，默认为 softmax)
    """

    def __init__(self, attn_dropout=0.0):
        self.dropout = Dropout(attn_dropout)
        self.activation = Activation("softmax")

    def __call__(self, q, k, v, mask):
        """应用缩放点积注意力。

        参数:
          q: 查询
          k: 键
          v: 值
          mask: 如果需要，进行掩码 -- 将 softmax 设置为非常大的值

        返回:
          (层输出, 注意力权重) 的元组
        """
        temper = tf.sqrt(tf.cast(tf.shape(k)[-1], dtype="float32"))
        attn = Lambda(lambda x: K.batch_dot(x[0], x[1], axes=[2, 2]) / temper)([q, k])  # shape=(batch, q, k)
        if mask is not None:
            mmask = Lambda(lambda x: (-1e9) * (1.0 - K.cast(x, "float32")))(mask)  # 设置为无穷大
            attn = Add()([attn, mmask])
        attn = self.activation(attn)
        attn = self.dropout(attn)
        output = Lambda(lambda x: K.batch_dot(x[0], x[1]))([attn, v])
        return output, attn


class InterpretableMultiHeadAttention:
    """定义可解释的多头注意力层。

    属性:
      n_head: 头数
      d_k: 每个头的键/查询维度
      d_v: 值维度
      dropout: 要应用的 dropout 率
      qs_layers: 跨头的查询列表
      ks_layers: 跨头的键列表
      vs_layers: 跨头的值列表
      attention: 缩放点积注意力层
      w_o: 输出权重矩阵，用于将内部状态投影到原始 TFT 状态大小
    """

    def __init__(self, n_head, d_model, dropout):
        """初始化层。

        参数:
          n_head: 头数
          d_model: TFT 状态维度
          dropout: Dropout 丢弃率
        """

        self.n_head = n_head
        self.d_k = self.d_v = d_k = d_v = d_model // n_head
        self.dropout = dropout

        self.qs_layers = []
        self.ks_layers = []
        self.vs_layers = []

        # 使用相同的值层以方便解释
        vs_layer = Dense(d_v, use_bias=False)

        for _ in range(n_head):
            self.qs_layers.append(Dense(d_k, use_bias=False))
            self.ks_layers.append(Dense(d_k, use_bias=False))
            self.vs_layers.append(vs_layer)  # 使用相同的 vs_layer

        self.attention = ScaledDotProductAttention()
        self.w_o = Dense(d_model, use_bias=False)

    def __call__(self, q, k, v, mask=None):
        """应用可解释的多头注意力。

        使用 T 表示输入到 transformer 的时间步数。

        参数:
          q: 查询张量，形状=(?, T, d_model)
          k: 键，形状=(?, T, d_model)
          v: 值，形状=(?, T, d_model)
          mask: 如果需要，进行掩码，形状=(?, T, T)

        返回:
          (层输出, 注意力权重) 的元组
        """
        n_head = self.n_head

        heads = []
        attns = []
        for i in range(n_head):
            qs = self.qs_layers[i](q)
            ks = self.ks_layers[i](k)
            vs = self.vs_layers[i](v)
            head, attn = self.attention(qs, ks, vs, mask)

            head_dropout = Dropout(self.dropout)(head)
            heads.append(head_dropout)
            attns.append(attn)
        head = K.stack(heads) if n_head > 1 else heads[0]
        attn = K.stack(attns)

        outputs = K.mean(head, axis=0) if n_head > 1 else head
        outputs = self.w_o(outputs)
        outputs = Dropout(self.dropout)(outputs)  # 输出 dropout

        return outputs, attn


class TFTDataCache:
    """缓存 TFT 的数据。"""

    _data_cache = {}

    @classmethod
    def update(cls, data, key):
        """更新缓存的数据。

        参数:
          data: 要更新的源
          key: 字典位置的键
        """
        cls._data_cache[key] = data

    @classmethod
    def get(cls, key):
        """返回存储在键位置的数据。"""
        return cls._data_cache[key].copy()

    @classmethod
    def contains(cls, key):
        """返回一个布尔值，指示缓存中是否存在该键。"""

        return key in cls._data_cache


# TFT 模型定义。
class TemporalFusionTransformer:
    """定义时序融合 Transformer。

    属性:
      name: 模型名称
      time_steps: 每个预测日期的总输入时间步数 (即时序融合解码器的宽度 N)
      input_size: 总输入数
      output_size: 总输出数
      category_counts: 每个分类变量的类别数
      n_multiprocessing_workers: 用于并行计算的工作进程数
      column_definition: 定义每列的 (字符串, 数据类型, 输入类型) 元组列表
      quantiles: TFT 预测的分位数
      use_cudnn: 是否使用 Keras CuDNNLSTM 或标准 LSTM 层
      hidden_layer_size: TFT 的内部状态大小
      dropout_rate: Dropout 丢弃率
      max_gradient_norm: 梯度裁剪的最大范数
      learning_rate: ADAM 优化器的初始学习率
      minibatch_size: 训练的小批量大小
      num_epochs: 训练的最大轮数
      early_stopping_patience: 在早停生效前，无改善的最大迭代次数
      num_encoder_steps: LSTM 编码器的大小 -- 即在预测日期前要使用的过去时间步数
      num_stacks: 要应用的自注意力层数 (对于基本 TFT，默认为 1)
      num_heads: 可解释多头注意力的头数
      model: TFT 的 Keras 模型
    """

    def __init__(self, raw_params, use_cudnn=False):
        """从参数构建 TFT。

        参数:
          raw_params: 定义 TFT 的参数
          use_cudnn: 是否使用 CUDNN GPU 优化的 LSTM
        """

        self.name = self.__class__.__name__

        params = dict(raw_params)  # 本地复制

        # 数据参数
        self.time_steps = int(params["total_time_steps"])
        self.input_size = int(params["input_size"])
        self.output_size = int(params["output_size"])
        self.category_counts = json.loads(str(params["category_counts"]))
        self.n_multiprocessing_workers = int(params["multiprocessing_workers"])

        # TFT 的相关索引
        self._input_obs_loc = json.loads(str(params["input_obs_loc"]))
        self._static_input_loc = json.loads(str(params["static_input_loc"]))
        self._known_regular_input_idx = json.loads(str(params["known_regular_inputs"]))
        self._known_categorical_input_idx = json.loads(str(params["known_categorical_inputs"]))

        self.column_definition = params["column_definition"]

        # 网络参数
        self.quantiles = [0.1, 0.5, 0.9]
        self.use_cudnn = use_cudnn  # 是否使用 GPU 优化的 LSTM
        self.hidden_layer_size = int(params["hidden_layer_size"])
        self.dropout_rate = float(params["dropout_rate"])
        self.max_gradient_norm = float(params["max_gradient_norm"])
        self.learning_rate = float(params["learning_rate"])
        self.minibatch_size = int(params["minibatch_size"])
        self.num_epochs = int(params["num_epochs"])
        self.early_stopping_patience = int(params["early_stopping_patience"])

        self.num_encoder_steps = int(params["num_encoder_steps"])
        self.num_stacks = int(params["stack_size"])
        self.num_heads = int(params["num_heads"])

        # 序列化选项
        self._temp_folder = os.path.join(params["model_folder"], "tmp")
        self.reset_temp_folder()

        # 用于存储 Tensorflow 节点以进行注意力计算的额外组件
        self._input_placeholder = None
        self._attention_components = None
        self._prediction_parts = None

        print("*** {} 参数 ***".format(self.name))
        for k in params:
            print("# {} = {}".format(k, params[k]))

        # 构建模型
        self.model = self.build_model()

    def get_tft_embeddings(self, all_inputs):
        """将原始输入转换为嵌入。

        对连续变量应用线性变换，并对分类变量使用嵌入。

        参数:
          all_inputs: 要转换的输入

        返回:
          转换后输入的张量。
        """

        time_steps = self.time_steps

        # 健全性检查
        for i in self._known_regular_input_idx:
            if i in self._input_obs_loc:
                raise ValueError("观测不能是先验已知的！")
        for i in self._input_obs_loc:
            if i in self._static_input_loc:
                raise ValueError("观测不能是静态的！")

        if all_inputs.get_shape().as_list()[-1] != self.input_size:
            raise ValueError(
                "输入数量非法！观察到的输入={}, 预期={}".format(
                    all_inputs.get_shape().as_list()[-1], self.input_size
                )
            )

        num_categorical_variables = len(self.category_counts)
        num_regular_variables = self.input_size - num_categorical_variables

        embedding_sizes = [self.hidden_layer_size for i, size in enumerate(self.category_counts)]

        embeddings = []
        for i in range(num_categorical_variables):
            embedding = tf.keras.Sequential(
                [
                    tf.keras.layers.InputLayer([time_steps]),
                    tf.keras.layers.Embedding(
                        self.category_counts[i], embedding_sizes[i], input_length=time_steps, dtype=tf.float32
                    ),
                ]
            )
            embeddings.append(embedding)

        regular_inputs, categorical_inputs = (
            all_inputs[:, :, :num_regular_variables],
            all_inputs[:, :, num_regular_variables:],
        )

        embedded_inputs = [embeddings[i](categorical_inputs[Ellipsis, i]) for i in range(num_categorical_variables)]

        # 静态输入
        if self._static_input_loc:
            static_inputs = [
                tf.keras.layers.Dense(self.hidden_layer_size)(regular_inputs[:, 0, i : i + 1])
                for i in range(num_regular_variables)
                if i in self._static_input_loc
            ] + [
                embedded_inputs[i][:, 0, :]
                for i in range(num_categorical_variables)
                if i + num_regular_variables in self._static_input_loc
            ]
            static_inputs = tf.keras.backend.stack(static_inputs, axis=1)

        else:
            static_inputs = None

        def convert_real_to_embedding(x):
            """对时变输入应用线性变换。"""
            return tf.keras.layers.TimeDistributed(tf.keras.layers.Dense(self.hidden_layer_size))(x)

        # 目标
        obs_inputs = tf.keras.backend.stack(
            [convert_real_to_embedding(regular_inputs[Ellipsis, i : i + 1]) for i in self._input_obs_loc], axis=-1
        )

        # 观察到的（先验未知）输入
        wired_embeddings = []
        for i in range(num_categorical_variables):
            if i not in self._known_categorical_input_idx and i + num_regular_variables not in self._input_obs_loc:
                e = embeddings[i](categorical_inputs[:, :, i])
                wired_embeddings.append(e)

        unknown_inputs = []
        for i in range(regular_inputs.shape[-1]):
            if i not in self._known_regular_input_idx and i not in self._input_obs_loc:
                e = convert_real_to_embedding(regular_inputs[Ellipsis, i : i + 1])
                unknown_inputs.append(e)

        if unknown_inputs + wired_embeddings:
            unknown_inputs = tf.keras.backend.stack(unknown_inputs + wired_embeddings, axis=-1)
        else:
            unknown_inputs = None

        # 先验已知的输入
        known_regular_inputs = [
            convert_real_to_embedding(regular_inputs[Ellipsis, i : i + 1])
            for i in self._known_regular_input_idx
            if i not in self._static_input_loc
        ]
        known_categorical_inputs = [
            embedded_inputs[i]
            for i in self._known_categorical_input_idx
            if i + num_regular_variables not in self._static_input_loc
        ]

        known_combined_layer = tf.keras.backend.stack(known_regular_inputs + known_categorical_inputs, axis=-1)

        return unknown_inputs, known_combined_layer, obs_inputs, static_inputs

    def _get_single_col_by_type(self, input_type):
        """返回输入类型的单列名称。"""

        return utils.get_single_col_by_input_type(input_type, self.column_definition)

    def training_data_cached(self):
        """返回一个布尔值，指示是否已缓存训练数据。"""

        return TFTDataCache.contains("train") and TFTDataCache.contains("valid")

    def cache_batched_data(self, data, cache_key, num_samples=-1):
        """批量处理并缓存数据一次，以在训练期间使用。

        参数:
          data: 要批量处理和缓存的数据
          cache_key: 用于缓存的键
          num_samples: 要提取的最大样本数（-1 表示使用所有数据）
        """

        if num_samples > 0:
            TFTDataCache.update(self._batch_sampled_data(data, max_samples=num_samples), cache_key)
        else:
            TFTDataCache.update(self._batch_data(data), cache_key)

        print('缓存的数据 "{}" 已更新'.format(cache_key))

    def _batch_sampled_data(self, data, max_samples):
        """将分段采样为兼容的格式。

        参数:
          data: 要采样和批量处理的源数据
          max_samples: 批次中的最大样本数

        返回:
          具有指定最大样本数的批量数据的字典。
        """

        if max_samples < 1:
            raise ValueError("指定的样本数非法！样本数={}".format(max_samples))

        id_col = self._get_single_col_by_type(InputTypes.ID)
        time_col = self._get_single_col_by_type(InputTypes.TIME)

        data.sort_values(by=[id_col, time_col], inplace=True)

        print("获取有效的采样位置。")
        valid_sampling_locations = []
        split_data_map = {}
        for identifier, df in data.groupby(id_col, group_key=False):
            print("正在获取 {} 的位置".format(identifier))
            num_entries = len(df)
            if num_entries >= self.time_steps:
                valid_sampling_locations += [
                    (identifier, self.time_steps + i) for i in range(num_entries - self.time_steps + 1)
                ]
            split_data_map[identifier] = df

        inputs = np.zeros((max_samples, self.time_steps, self.input_size))
        outputs = np.zeros((max_samples, self.time_steps, self.output_size))
        time = np.empty((max_samples, self.time_steps, 1), dtype=object)
        identifiers = np.empty((max_samples, self.time_steps, 1), dtype=object)

        if max_samples > 0 and len(valid_sampling_locations) > max_samples:
            print("正在提取 {} 个样本...".format(max_samples))
            ranges = [
                valid_sampling_locations[i]
                for i in np.random.choice(len(valid_sampling_locations), max_samples, replace=False)
            ]
        else:
            print("最大样本数={} 超过了可用分段数={}".format(max_samples, len(valid_sampling_locations)))
            ranges = valid_sampling_locations

        id_col = self._get_single_col_by_type(InputTypes.ID)
        time_col = self._get_single_col_by_type(InputTypes.TIME)
        target_col = self._get_single_col_by_type(InputTypes.TARGET)
        input_cols = [tup[0] for tup in self.column_definition if tup[2] not in {InputTypes.ID, InputTypes.TIME}]

        for i, tup in enumerate(ranges):
            if (i + 1 % 1000) == 0:
                print(i + 1, "of", max_samples, "个样本已完成...")
            identifier, start_idx = tup
            sliced = split_data_map[identifier].iloc[start_idx - self.time_steps : start_idx]
            inputs[i, :, :] = sliced[input_cols]
            outputs[i, :, :] = sliced[[target_col]]
            time[i, :, 0] = sliced[time_col]
            identifiers[i, :, 0] = sliced[id_col]

        sampled_data = {
            "inputs": inputs,
            "outputs": outputs[:, self.num_encoder_steps :, :],
            "active_entries": np.ones_like(outputs[:, self.num_encoder_steps :, :]),
            "time": time,
            "identifier": identifiers,
        }

        return sampled_data

    def _batch_data(self, data):
        """批量处理数据以进行训练。

        将原始数据帧从二维表格格式转换为批处理的三维数组，以馈入 Keras 模型。

        参数:
          data: 要批量处理的 DataFrame

        返回:
          形状为 (?, self.time_steps, self.input_size) 的批处理 Numpy 数组
        """

        # 函数。
        def _batch_single_entity(input_data):
            time_steps = len(input_data)
            lags = self.time_steps
            x = input_data.values
            if time_steps >= lags:
                return np.stack([x[i : time_steps - (lags - 1) + i, :] for i in range(lags)], axis=1)

            else:
                return None

        id_col = self._get_single_col_by_type(InputTypes.ID)
        time_col = self._get_single_col_by_type(InputTypes.TIME)
        target_col = self._get_single_col_by_type(InputTypes.TARGET)
        input_cols = [tup[0] for tup in self.column_definition if tup[2] not in {InputTypes.ID, InputTypes.TIME}]

        data_map = {}
        for _, sliced in data.groupby(id_col, group_key=False):
            col_mappings = {"identifier": [id_col], "time": [time_col], "outputs": [target_col], "inputs": input_cols}

            for k in col_mappings:
                cols = col_mappings[k]
                arr = _batch_single_entity(sliced[cols].copy())

                if k not in data_map:
                    data_map[k] = [arr]
                else:
                    data_map[k].append(arr)

        # 合并所有数据
        for k in data_map:
            data_map[k] = np.concatenate([i for i in data_map[k] if i is not None], axis=0)

        # 缩短目标，以便我们只获得解码器步骤
        data_map["outputs"] = data_map["outputs"][:, self.num_encoder_steps :, :]

        active_entries = np.ones_like(data_map["outputs"])
        if "active_entries" not in data_map:
            data_map["active_entries"] = active_entries
        else:
            data_map["active_entries"].append(active_entries)

        return data_map

    def _get_active_locations(self, x):
        """格式化 Keras 训练的样本权重。"""
        return (np.sum(x, axis=-1) > 0.0) * 1.0

    def _build_base_graph(self):
        """返回定义 TFT 层的图。"""

        # 大小定义。
        time_steps = self.time_steps
        combined_input_size = self.input_size
        encoder_steps = self.num_encoder_steps

        # 输入。
        all_inputs = tf.keras.layers.Input(
            shape=(
                time_steps,
                combined_input_size,
            )
        )

        unknown_inputs, known_combined_layer, obs_inputs, static_inputs = self.get_tft_embeddings(all_inputs)

        # 隔离已知和观察到的历史输入。
        if unknown_inputs is not None:
            historical_inputs = concat(
                [
                    unknown_inputs[:, :encoder_steps, :],
                    known_combined_layer[:, :encoder_steps, :],
                    obs_inputs[:, :encoder_steps, :],
                ],
                axis=-1,
            )
        else:
            historical_inputs = concat(
                [known_combined_layer[:, :encoder_steps, :], obs_inputs[:, :encoder_steps, :]], axis=-1
            )

        # 仅隔离已知的未来输入。
        future_inputs = known_combined_layer[:, encoder_steps:, :]

        def static_combine_and_mask(embedding):
            """将变量选择网络应用于静态输入。

            参数:
              embedding: 转换后的静态输入

            返回:
              变量选择网络的张量输出
            """

            # 添加时间特征
            _, num_static, _ = embedding.get_shape().as_list()

            flatten = tf.keras.layers.Flatten()(embedding)

            # 使用门控残差网络进行非线性变换。
            mlp_outputs = gated_residual_network(
                flatten,
                self.hidden_layer_size,
                output_size=num_static,
                dropout_rate=self.dropout_rate,
                use_time_distributed=False,
                additional_context=None,
            )

            sparse_weights = tf.keras.layers.Activation("softmax")(mlp_outputs)
            sparse_weights = K.expand_dims(sparse_weights, axis=-1)

            trans_emb_list = []
            for i in range(num_static):
                e = gated_residual_network(
                    embedding[:, i : i + 1, :],
                    self.hidden_layer_size,
                    dropout_rate=self.dropout_rate,
                    use_time_distributed=False,
                )
                trans_emb_list.append(e)

            transformed_embedding = concat(trans_emb_list, axis=1)

            combined = tf.keras.layers.Multiply()([sparse_weights, transformed_embedding])

            static_vec = K.sum(combined, axis=1)

            return static_vec, sparse_weights

        static_encoder, static_weights = static_combine_and_mask(static_inputs)

        static_context_variable_selection = gated_residual_network(
            static_encoder, self.hidden_layer_size, dropout_rate=self.dropout_rate, use_time_distributed=False
        )
        static_context_enrichment = gated_residual_network(
            static_encoder, self.hidden_layer_size, dropout_rate=self.dropout_rate, use_time_distributed=False
        )
        static_context_state_h = gated_residual_network(
            static_encoder, self.hidden_layer_size, dropout_rate=self.dropout_rate, use_time_distributed=False
        )
        static_context_state_c = gated_residual_network(
            static_encoder, self.hidden_layer_size, dropout_rate=self.dropout_rate, use_time_distributed=False
        )

        def lstm_combine_and_mask(embedding):
            """应用时间变量选择网络。

            参数:
              embedding: 转换后的输入。

            返回:
              处理后的张量输出。
            """

            # 添加时间特征
            _, time_steps, embedding_dim, num_inputs = embedding.get_shape().as_list()

            flatten = K.reshape(embedding, [-1, time_steps, embedding_dim * num_inputs])

            expanded_static_context = K.expand_dims(static_context_variable_selection, axis=1)

            # 变量选择权重
            mlp_outputs, static_gate = gated_residual_network(
                flatten,
                self.hidden_layer_size,
                output_size=num_inputs,
                dropout_rate=self.dropout_rate,
                use_time_distributed=True,
                additional_context=expanded_static_context,
                return_gate=True,
            )

            sparse_weights = tf.keras.layers.Activation("softmax")(mlp_outputs)
            sparse_weights = tf.expand_dims(sparse_weights, axis=2)

            # 非线性处理和权重应用
            trans_emb_list = []
            for i in range(num_inputs):
                grn_output = gated_residual_network(
                    embedding[Ellipsis, i],
                    self.hidden_layer_size,
                    dropout_rate=self.dropout_rate,
                    use_time_distributed=True,
                )
                trans_emb_list.append(grn_output)

            transformed_embedding = stack(trans_emb_list, axis=-1)

            combined = tf.keras.layers.Multiply()([sparse_weights, transformed_embedding])
            temporal_ctx = K.sum(combined, axis=-1)

            return temporal_ctx, sparse_weights, static_gate

        historical_features, historical_flags, _ = lstm_combine_and_mask(historical_inputs)
        future_features, future_flags, _ = lstm_combine_and_mask(future_inputs)

        # LSTM 层
        def get_lstm(return_state):
            """返回使用默认参数初始化的 LSTM 单元。"""
            if self.use_cudnn:
                lstm = tf.keras.layers.CuDNNLSTM(
                    self.hidden_layer_size,
                    return_sequences=True,
                    return_state=return_state,
                    stateful=False,
                )
            else:
                lstm = tf.keras.layers.LSTM(
                    self.hidden_layer_size,
                    return_sequences=True,
                    return_state=return_state,
                    stateful=False,
                    activation="tanh",
                    recurrent_activation="sigmoid",
                    recurrent_dropout=0,
                    unroll=False,
                    use_bias=True,
                )
            return lstm

        history_lstm, state_h, state_c = get_lstm(return_state=True)(
            historical_features, initial_state=[static_context_state_h, static_context_state_c]
        )

        future_lstm = get_lstm(return_state=False)(future_features, initial_state=[state_h, state_c])

        lstm_layer = concat([history_lstm, future_lstm], axis=1)

        # 应用门控跳跃连接
        input_embeddings = concat([historical_features, future_features], axis=1)

        lstm_layer, _ = apply_gating_layer(lstm_layer, self.hidden_layer_size, self.dropout_rate, activation=None)
        temporal_feature_layer = add_and_norm([lstm_layer, input_embeddings])

        # 静态丰富层
        expanded_static_context = K.expand_dims(static_context_enrichment, axis=1)
        enriched, _ = gated_residual_network(
            temporal_feature_layer,
            self.hidden_layer_size,
            dropout_rate=self.dropout_rate,
            use_time_distributed=True,
            additional_context=expanded_static_context,
            return_gate=True,
        )

        # 解码器自注意力
        self_attn_layer = InterpretableMultiHeadAttention(
            self.num_heads, self.hidden_layer_size, dropout=self.dropout_rate
        )

        mask = get_decoder_mask(enriched)
        x, self_att = self_attn_layer(enriched, enriched, enriched, mask=mask)

        x, _ = apply_gating_layer(x, self.hidden_layer_size, dropout_rate=self.dropout_rate, activation=None)
        x = add_and_norm([x, enriched])

        # 输出的非线性处理
        decoder = gated_residual_network(
            x, self.hidden_layer_size, dropout_rate=self.dropout_rate, use_time_distributed=True
        )

        # 最终跳跃连接
        decoder, _ = apply_gating_layer(decoder, self.hidden_layer_size, activation=None)
        transformer_layer = add_and_norm([decoder, temporal_feature_layer])

        # 用于可解释性的注意力组件
        attention_components = {
            # 时间注意力权重
            "decoder_self_attn": self_att,
            # 静态变量选择权重
            "static_flags": static_weights[Ellipsis, 0],
            # 过去输入的变量选择权重
            "historical_flags": historical_flags[Ellipsis, 0, :],
            # 未来输入的变量选择权重
            "future_flags": future_flags[Ellipsis, 0, :],
        }

        return transformer_layer, all_inputs, attention_components

    def build_model(self):
        """构建模型并定义训练损失。

        返回:
          完全定义的 Keras 模型。
        """

        with tf.variable_scope(self.name):
            transformer_layer, all_inputs, attention_components = self._build_base_graph()

            outputs = tf.keras.layers.TimeDistributed(tf.keras.layers.Dense(self.output_size * len(self.quantiles)))(
                transformer_layer[Ellipsis, self.num_encoder_steps :, :]
            )

            self._attention_components = attention_components

            adam = tf.keras.optimizers.Adam(lr=self.learning_rate, clipnorm=self.max_gradient_norm)

            model = tf.keras.Model(inputs=all_inputs, outputs=outputs)

            print(model.summary())

            valid_quantiles = self.quantiles
            output_size = self.output_size

            class QuantileLossCalculator:
                """计算预定分位数的组合分位数损失。

                属性:
                  quantiles: 用于计算损失的分位数
                """

                def __init__(self, quantiles):
                    """使用用于损失计算的分位数初始化计算机。

                    参数:
                      quantiles: 用于计算的分位数。
                    """
                    self.quantiles = quantiles

                def quantile_loss(self, a, b):
                    """返回指定分位数的分位数损失。

                    参数:
                      a: 目标
                      b: 预测
                    """
                    quantiles_used = set(self.quantiles)

                    loss = 0.0
                    for i, quantile in enumerate(valid_quantiles):
                        if quantile in quantiles_used:
                            loss += utils.tensorflow_quantile_loss(
                                a[Ellipsis, output_size * i : output_size * (i + 1)],
                                b[Ellipsis, output_size * i : output_size * (i + 1)],
                                quantile,
                            )
                    return loss

            quantile_loss = QuantileLossCalculator(valid_quantiles).quantile_loss

            model.compile(loss=quantile_loss, optimizer=adam, sample_weight_mode="temporal")

            self._input_placeholder = all_inputs

        return model

    def fit(self, train_df=None, valid_df=None):
        """为给定的训练和验证数据拟合深度神经网络。

        参数:
          train_df: 训练数据的 DataFrame
          valid_df: 验证数据的 DataFrame
        """

        print("*** 正在拟合 {} ***".format(self.name))

        # 添加相关回调
        callbacks = [
            tf.keras.callbacks.EarlyStopping(monitor="val_loss", patience=self.early_stopping_patience, min_delta=1e-4),
            tf.keras.callbacks.ModelCheckpoint(
                filepath=self.get_keras_saved_path(self._temp_folder),
                monitor="val_loss",
                save_best_only=True,
                save_weights_only=True,
            ),
            tf.keras.callbacks.TerminateOnNaN(),
        ]

        print("正在获取 batched_data")
        if train_df is None:
            print("正在使用缓存的训练数据")
            train_data = TFTDataCache.get("train")
        else:
            train_data = self._batch_data(train_df)

        if valid_df is None:
            print("正在使用缓存的验证数据")
            valid_data = TFTDataCache.get("valid")
        else:
            valid_data = self._batch_data(valid_df)

        print("正在使用 keras standard fit")

        def _unpack(data):
            return data["inputs"], data["outputs"], self._get_active_locations(data["active_entries"])

        data, labels, active_flags = _unpack(train_data)
        val_data, val_labels, val_flags = _unpack(valid_data)

        all_callbacks = callbacks

        self.model.fit(
            x=data,
            y=np.concatenate([labels, labels, labels], axis=-1),
            sample_weight=active_flags,
            epochs=self.num_epochs,
            batch_size=self.minibatch_size,
            validation_data=(val_data, np.concatenate([val_labels, val_labels, val_labels], axis=-1), val_flags),
            callbacks=all_callbacks,
            shuffle=True,
            use_multiprocessing=True,
            workers=self.n_multiprocessing_workers,
        )

        # 再次加载最佳检查点
        tmp_checkpont = self.get_keras_saved_path(self._temp_folder)
        if os.path.exists(tmp_checkpont):
            self.load(self._temp_folder, use_keras_loadings=True)

        else:
            print("无法从 {} 加载，正在跳过 ...".format(self._temp_folder))

    def evaluate(self, data=None, eval_metric="loss"):
        """将评估指标应用于训练数据。

        参数:
          data: 用于评估的数据帧
          eval_metric: 要返回的评估指标，基于模型定义。

        返回:
          计算出的评估损失。
        """

        if data is None:
            print("正在使用缓存的验证数据")
            raw_data = TFTDataCache.get("valid")
        else:
            raw_data = self._batch_data(data)

        inputs = raw_data["inputs"]
        outputs = raw_data["outputs"]
        active_entries = self._get_active_locations(raw_data["active_entries"])

        metric_values = self.model.evaluate(
            x=inputs,
            y=np.concatenate([outputs, outputs, outputs], axis=-1),
            sample_weight=active_entries,
            workers=16,
            use_multiprocessing=True,
        )

        metrics = pd.Series(metric_values, self.model.metrics_names)

        return metrics[eval_metric]

    def predict(self, df, return_targets=False):
        """为给定的输入数据集计算预测。

        参数:
          df: 输入数据帧
          return_targets: 是否也返回与预测对齐的输出以方便评估

        返回:
          输入数据帧或 (输入数据帧, 对齐的输出数据帧) 的元组。
        """

        data = self._batch_data(df)

        inputs = data["inputs"]
        time = data["time"]
        identifier = data["identifier"]
        outputs = data["outputs"]

        combined = self.model.predict(inputs, workers=16, use_multiprocessing=True, batch_size=self.minibatch_size)

        if self.output_size != 1:
            raise NotImplementedError("当前版本仅支持一维目标！")

        def format_outputs(prediction):
            """返回格式化的预测数据帧。"""

            flat_prediction = pd.DataFrame(
                prediction[:, :, 0], columns=["t+{}".format(i) for i in range(self.time_steps - self.num_encoder_steps)]
            )
            cols = list(flat_prediction.columns)
            flat_prediction["forecast_time"] = time[:, self.num_encoder_steps - 1, 0]
            flat_prediction["identifier"] = identifier[:, 0, 0]

            return flat_prediction[["forecast_time", "identifier"] + cols]

        process_map = {
            "p{}".format(int(q * 100)): combined[Ellipsis, i * self.output_size : (i + 1) * self.output_size]
            for i, q in enumerate(self.quantiles)
        }

        if return_targets:
            process_map["targets"] = outputs

        return {k: format_outputs(process_map[k]) for k in process_map}

    def get_attention(self, df):
        """为给定数据集计算 TFT 注意力权重。

        参数:
          df: 输入数据帧

        返回:
            一个包含时间注意力权重和变量选择权重的 numpy 数组的字典，以及它们的标识符和时间索引
        """

        data = self._batch_data(df)
        inputs = data["inputs"]
        identifiers = data["identifier"]
        time = data["time"]

        def get_batch_attention_weights(input_batch):
            """返回给定小批量数据的权重。"""
            input_placeholder = self._input_placeholder
            attention_weights = {}
            for k in self._attention_components:
                attention_weight = tf.keras.backend.get_session().run(
                    self._attention_components[k], {input_placeholder: input_batch.astype(np.float32)}
                )
                attention_weights[k] = attention_weight
            return attention_weights

        batch_size = self.minibatch_size
        n = inputs.shape[0]
        num_batches = n // batch_size
        if n - (num_batches * batch_size) > 0:
            num_batches += 1

        batched_inputs = [inputs[i * batch_size : (i + 1) * batch_size, Ellipsis] for i in range(num_batches)]

        attention_by_batch = [get_batch_attention_weights(batch) for batch in batched_inputs]
        attention_weights = {}
        for k in self._attention_components:
            attention_weights[k] = []
            for batch_weights in attention_by_batch:
                attention_weights[k].append(batch_weights[k])

            if len(attention_weights[k][0].shape) == 4:
                tmp = np.concatenate(attention_weights[k], axis=1)
            else:
                tmp = np.concatenate(attention_weights[k], axis=0)

            del attention_weights[k]
            gc.collect()
            attention_weights[k] = tmp

        attention_weights["identifiers"] = identifiers[:, 0, 0]
        attention_weights["time"] = time[:, :, 0]

        return attention_weights

    # 序列化。
    def reset_temp_folder(self):
        """删除并重新创建包含临时 Keras 训练输出的文件夹。"""
        print("正在重置临时文件夹...")
        utils.create_folder_if_not_exist(self._temp_folder)
        shutil.rmtree(self._temp_folder)
        os.makedirs(self._temp_folder)

    def get_keras_saved_path(self, model_folder):
        """返回 keras 检查点的路径。"""
        return os.path.join(model_folder, "{}.check".format(self.name))

    def save(self, model_folder):
        """保存最佳 TFT 权重。

        参数:
          model_folder: 序列化模型的位置。
        """
        utils.save(tf.keras.backend.get_session(), model_folder, cp_name=self.name, scope=self.name)

    def load(self, model_folder, use_keras_loadings=False):
        """加载 TFT 权重。

        参数:
          model_folder: 包含序列化模型的文件夹。
          use_keras_loadings: 是否从 Keras 检查点加载。

        返回:

        """
        if use_keras_loadings:
            serialisation_path = self.get_keras_saved_path(model_folder)
            print("正在从 {} 加载模型".format(serialisation_path))
            self.model.load_weights(serialisation_path)
        else:
            utils.load(tf.keras.backend.get_session(), model_folder, cp_name=self.name, scope=self.name)

    @classmethod
    def get_hyperparm_choices(cls):
        """返回用于随机搜索的超参数范围。"""
        return {
            "dropout_rate": [0.1, 0.2, 0.3, 0.4, 0.5, 0.7, 0.9],
            "hidden_layer_size": [10, 20, 40, 80, 160, 240, 320],
            "minibatch_size": [64, 128, 256],
            "learning_rate": [1e-4, 1e-3, 1e-2],
            "max_gradient_norm": [0.01, 1.0, 100.0],
            "num_heads": [1, 4],
            "stack_size": [1],
        }
