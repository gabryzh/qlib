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
"""代码库中使用的通用辅助函数。"""
import os
import pathlib

import numpy as np
import tensorflow as tf
from tensorflow.python.tools.inspect_checkpoint import print_tensors_in_checkpoint_file


# 通用。
def get_single_col_by_input_type(input_type, column_definition):
    """返回单列的名称。

    参数:
      input_type: 要提取的列的输入类型
      column_definition: 实验的列定义列表
    """

    l = [tup[0] for tup in column_definition if tup[2] == input_type]

    if len(l) != 1:
        raise ValueError("列数无效 {}".format(input_type))

    return l[0]


def extract_cols_from_data_type(data_type, column_definition, excluded_input_types):
    """提取与定义的数据类型相对应的列的名称。

    参数:
      data_type: 要提取的列的数据类型。
      column_definition: 要使用的列定义。
      excluded_input_types: 要排除的输入类型集

    返回:
      具有指定数据类型的列的名称列表。
    """
    return [tup[0] for tup in column_definition if tup[1] == data_type and tup[2] not in excluded_input_types]


# 损失函数。
def tensorflow_quantile_loss(y, y_pred, quantile):
    """计算 tensorflow 的分位数损失。

    TFT 主论文“训练过程”部分定义的标准分位数损失

    参数:
      y: 目标
      y_pred: 预测
      quantile: 用于损失计算的分位数（介于 0 和 1 之间）

    返回:
      分位数损失的张量。
    """

    # 检查分位数
    if quantile < 0 or quantile > 1:
        raise ValueError("非法分位数={}！值应介于 0 和 1 之间。".format(quantile))

    prediction_underflow = y - y_pred
    q_loss = quantile * tf.maximum(prediction_underflow, 0.0) + (1.0 - quantile) * tf.maximum(
        -prediction_underflow, 0.0
    )

    return tf.reduce_sum(q_loss, axis=-1)


def numpy_normalised_quantile_loss(y, y_pred, quantile):
    """计算 numpy 数组的归一化分位数损失。

    使用 TFT 主论文“训练过程”部分定义的 q-Risk 指标。

    参数:
      y: 目标
      y_pred: 预测
      quantile: 用于损失计算的分位数（介于 0 和 1 之间）

    返回:
      归一化分位数损失的浮点数。
    """
    prediction_underflow = y - y_pred
    weighted_errors = quantile * np.maximum(prediction_underflow, 0.0) + (1.0 - quantile) * np.maximum(
        -prediction_underflow, 0.0
    )

    quantile_loss = weighted_errors.mean()
    normaliser = y.abs().mean()

    return 2 * quantile_loss / normaliser


# OS 相关函数。
def create_folder_if_not_exist(directory):
    """如果文件夹不存在，则创建文件夹。

    参数:
      directory: 要创建的文件夹路径。
    """
    # 也递归创建目录
    pathlib.Path(directory).mkdir(parents=True, exist_ok=True)


# Tensorflow 相关函数。
def get_default_tensorflow_config(tf_device="gpu", gpu_id=0):
    """创建用于在 CPU 或 GPU 上运行的图的 tensorflow 配置。

    指定是在 gpu 还是 cpu 上运行图，以及在多 GPU 机器上使用哪个 GPU ID。

    参数:
      tf_device: 'cpu' 或 'gpu'
      gpu_id: 如果相关，要使用的 GPU ID

    返回:
      Tensorflow 配置。
    """

    if tf_device == "cpu":
        os.environ["CUDA_VISIBLE_DEVICES"] = "-1"  # 用于在 cpu 上训练
        tf_config = tf.ConfigProto(log_device_placement=False, device_count={"GPU": 0})

    else:
        os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
        os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu_id)

        print("正在选择 GPU ID={}".format(gpu_id))

        tf_config = tf.ConfigProto(log_device_placement=False)
        tf_config.gpu_options.allow_growth = True

    return tf_config


def save(tf_session, model_folder, cp_name, scope=None):
    """将 Tensorflow 图保存到检查点。

    将给定变量范围下的所有可训练变量保存到检查点。

    参数:
      tf_session: 包含图的会话
      model_folder: 保存模型的文件夹
      cp_name: Tensorflow 检查点的名称
      scope: 包含要保存的变量的变量范围
    """
    # 保存模型
    if scope is None:
        saver = tf.train.Saver()
    else:
        var_list = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope=scope)
        saver = tf.train.Saver(var_list=var_list, max_to_keep=100000)

    save_path = saver.save(tf_session, os.path.join(model_folder, "{0}.ckpt".format(cp_name)))
    print("模型已保存到: {0}".format(save_path))


def load(tf_session, model_folder, cp_name, scope=None, verbose=False):
    """从检查点加载 Tensorflow 图。

    参数:
      tf_session: 要加载图的会话
      model_folder: 包含序列化模型的文件夹
      cp_name: Tensorflow 检查点的名称
      scope: 要使用的变量范围。
      verbose: 是否打印其他调试信息。
    """
    # 加载模型
    load_path = os.path.join(model_folder, "{0}.ckpt".format(cp_name))

    print("正在从 {0} 加载模型".format(load_path))

    print_weights_in_checkpoint(model_folder, cp_name)

    initial_vars = set([v.name for v in tf.get_default_graph().as_graph_def().node])

    # Saver
    if scope is None:
        saver = tf.train.Saver()
    else:
        var_list = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope=scope)
        saver = tf.train.Saver(var_list=var_list, max_to_keep=100000)
    # 加载
    saver.restore(tf_session, load_path)
    all_vars = set([v.name for v in tf.get_default_graph().as_graph_def().node])

    if verbose:
        print("已恢复 {0}".format(",".join(initial_vars.difference(all_vars))))
        print("已存在 {0}".format(",".join(all_vars.difference(initial_vars))))
        print("全部 {0}".format(",".join(all_vars)))

    print("完成。")


def print_weights_in_checkpoint(model_folder, cp_name):
    """打印 Tensorflow 检查点中的所有权重。

    参数:
      model_folder: 包含检查点的文件夹
      cp_name: 检查点的名称

    返回:

    """
    load_path = os.path.join(model_folder, "{0}.ckpt".format(cp_name))

    print_tensors_in_checkpoint_file(file_name=load_path, tensor_name="", all_tensors=True, all_tensor_names=True)
