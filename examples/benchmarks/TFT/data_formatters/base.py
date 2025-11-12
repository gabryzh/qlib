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
"""实验的默认数据格式化函数。

对于新的数据集，继承自 GenericDataFormatter 并实现所有抽象函数。

这些特定于数据集的方法:
1) 定义模型使用的表格数据帧的列和输入类型
2) 执行必要的输入特征工程和归一化步骤
3) 还原预测的归一化
4) 负责训练、验证和测试的拆分

"""
import abc
import enum


# 类型定义
class DataTypes(enum.IntEnum):
    """定义每列的数值类型。"""

    REAL_VALUED = 0
    CATEGORICAL = 1
    DATE = 2


class InputTypes(enum.IntEnum):
    """定义每列的输入类型。"""

    TARGET = 0
    OBSERVED_INPUT = 1
    KNOWN_INPUT = 2
    STATIC_INPUT = 3
    ID = 4  # 用作实体标识符的单列
    TIME = 5  # 专门用作时间索引的单列


class GenericDataFormatter(abc.ABC):
    """所有数据格式化程序的抽象基类。

    用户可以实现下面的抽象方法来执行特定于数据集的操作。

    """

    @abc.abstractmethod
    def set_scalers(self, df):
        """使用提供的数据校准缩放器。"""
        raise NotImplementedError()

    @abc.abstractmethod
    def transform_inputs(self, df):
        """执行特征转换。"""
        raise NotImplementedError()

    @abc.abstractmethod
    def format_predictions(self, df):
        """还原任何归一化，以原始比例给出预测。"""
        raise NotImplementedError()

    @abc.abstractmethod
    def split_data(self, df):
        """执行默认的训练、验证和测试拆分。"""
        raise NotImplementedError()

    @property
    @abc.abstractmethod
    def _column_definition(self):
        """定义每列的顺序、输入类型和数据类型。"""
        raise NotImplementedError()

    @abc.abstractmethod
    def get_fixed_params(self):
        """定义模型用于训练的固定参数。

        需要以下键:
          'total_time_steps': 定义 TFT 使用的总时间步数
          'num_encoder_steps': 决定 LSTM 编码器的长度（即历史记录）
          'num_epochs': 训练的最大轮数
          'early_stopping_patience': keras 的早停参数
          'multiprocessing_workers': 用于数据处理的 cpu 数量

        返回:
          一个固定参数的字典，例如:

          fixed_params = {
              'total_time_steps': 252 + 5,
              'num_encoder_steps': 252,
              'num_epochs': 100,
              'early_stopping_patience': 5,
              'multiprocessing_workers': 5,
          }
        """
        raise NotImplementedError

    # 跨数据格式化程序的共享函数
    @property
    def num_classes_per_cat_input(self):
        """返回每个相关输入的类别数。

        这对于 keras 嵌入层是顺序必需的。
        """
        return self._num_classes_per_cat_input

    def get_num_samples_for_calibration(self):
        """获取默认的训练和验证样本数。

        用于对数据进行子采样以进行网络校准，值为 -1 表示使用所有可用样本。

        返回:
          (训练样本, 验证样本) 的元组
        """
        return -1, -1

    def get_column_definition(self):
        """按 TFT 预期的顺序列出格式化的列定义。"""

        column_definition = self._column_definition

        # 首先进行健全性检查。
        # 确保只存在一个 ID 和时间列
        def _check_single_column(input_type):
            length = len([tup for tup in column_definition if tup[2] == input_type])

            if length != 1:
                raise ValueError("类型 {} 的输入数量 ({}) 非法".format(input_type, length))

        _check_single_column(InputTypes.ID)
        _check_single_column(InputTypes.TIME)

        identifier = [tup for tup in column_definition if tup[2] == InputTypes.ID]
        time = [tup for tup in column_definition if tup[2] == InputTypes.TIME]
        real_inputs = [
            tup
            for tup in column_definition
            if tup[1] == DataTypes.REAL_VALUED and tup[2] not in {InputTypes.ID, InputTypes.TIME}
        ]
        categorical_inputs = [
            tup
            for tup in column_definition
            if tup[1] == DataTypes.CATEGORICAL and tup[2] not in {InputTypes.ID, InputTypes.TIME}
        ]

        return identifier + time + real_inputs + categorical_inputs

    def _get_input_columns(self):
        """返回所有输入列的名称。"""
        return [tup[0] for tup in self.get_column_definition() if tup[2] not in {InputTypes.ID, InputTypes.TIME}]

    def _get_tft_input_indices(self):
        """返回 TFT 所需的相关索引和输入大小。"""

        # 函数
        def _extract_tuples_from_data_type(data_type, defn):
            return [tup for tup in defn if tup[1] == data_type and tup[2] not in {InputTypes.ID, InputTypes.TIME}]

        def _get_locations(input_types, defn):
            return [i for i, tup in enumerate(defn) if tup[2] in input_types]

        # 开始提取
        column_definition = [
            tup for tup in self.get_column_definition() if tup[2] not in {InputTypes.ID, InputTypes.TIME}
        ]

        categorical_inputs = _extract_tuples_from_data_type(DataTypes.CATEGORICAL, column_definition)
        real_inputs = _extract_tuples_from_data_type(DataTypes.REAL_VALUED, column_definition)

        locations = {
            "input_size": len(self._get_input_columns()),
            "output_size": len(_get_locations({InputTypes.TARGET}, column_definition)),
            "category_counts": self.num_classes_per_cat_input,
            "input_obs_loc": _get_locations({InputTypes.TARGET}, column_definition),
            "static_input_loc": _get_locations({InputTypes.STATIC_INPUT}, column_definition),
            "known_regular_inputs": _get_locations({InputTypes.STATIC_INPUT, InputTypes.KNOWN_INPUT}, real_inputs),
            "known_categorical_inputs": _get_locations(
                {InputTypes.STATIC_INPUT, InputTypes.KNOWN_INPUT}, categorical_inputs
            ),
        }

        return locations

    def get_experiment_params(self):
        """返回实验的固定模型参数。"""

        required_keys = [
            "total_time_steps",
            "num_encoder_steps",
            "num_epochs",
            "early_stopping_patience",
            "multiprocessing_workers",
        ]

        fixed_params = self.get_fixed_params()

        for k in required_keys:
            if k not in fixed_params:
                raise ValueError("字段 {}".format(k) + " 在固定参数定义中缺失!")

        fixed_params["column_definition"] = self.get_column_definition()

        fixed_params.update(self._get_tft_input_indices())

        return fixed_params
