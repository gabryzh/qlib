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
"""用于超参数优化的类。

存在两个主要类:
1) HyperparamOptManager 用于在单台机器/GPU 上进行优化。
2) DistributedHyperparamOptManager 用于不同机器上的多个 GPU。
"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import collections
import os
import shutil
import libs.utils as utils
import numpy as np
import pandas as pd

Deque = collections.deque


class HyperparamOptManager:
    """使用随机搜索管理单个 GPU 的超参数优化。

    属性:
      param_ranges: 用于随机搜索的离散超参数范围。
      results: 验证结果的数据帧。
      fixed_params: 每个实验的固定模型参数。
      saved_params: 已训练参数的数据帧。
      best_score: 到目前为止观察到的最小验证损失。
      optimal_name: 最佳配置的键。
      hyperparam_folder: 保存优化输出的位置。
    """

    def __init__(self, param_ranges, fixed_params, model_folder, override_w_fixed_params=True):
        """实例化模型。

        参数:
          param_ranges: 用于随机搜索的离散超参数范围。
          fixed_params: 每个实验的固定模型参数。
          model_folder: 存储优化工件的文件夹。
          override_w_fixed_params: 是否用新的提供的值覆盖序列化的固定模型参数。
        """

        self.param_ranges = param_ranges

        self._max_tries = 1000
        self.results = pd.DataFrame()
        self.fixed_params = fixed_params
        self.saved_params = pd.DataFrame()

        self.best_score = np.Inf
        self.optimal_name = ""

        # 设置
        # 如果不存在，则创建用于保存的文件夹
        self.hyperparam_folder = model_folder
        utils.create_folder_if_not_exist(self.hyperparam_folder)

        self._override_w_fixed_params = override_w_fixed_params

    def load_results(self):
        """从先前的超参数优化中加载结果。

        返回:
          一个布尔值，指示是否可以加载先前结果。
        """
        print("正在从 {} 加载结果".format(self.hyperparam_folder))

        results_file = os.path.join(self.hyperparam_folder, "results.csv")
        params_file = os.path.join(self.hyperparam_folder, "params.csv")

        if os.path.exists(results_file) and os.path.exists(params_file):
            self.results = pd.read_csv(results_file, index_col=0)
            self.saved_params = pd.read_csv(params_file, index_col=0)

            if not self.results.empty:
                self.results.at["loss"] = self.results.loc["loss"].apply(float)
                self.best_score = self.results.loc["loss"].min()

                is_optimal = self.results.loc["loss"] == self.best_score
                self.optimal_name = self.results.T[is_optimal].index[0]

                return True

        return False

    def _get_params_from_name(self, name):
        """给定一个键，返回先前保存的参数。"""
        params = self.saved_params

        selected_params = dict(params[name])

        if self._override_w_fixed_params:
            for k in self.fixed_params:
                selected_params[k] = self.fixed_params[k]

        return selected_params

    def get_best_params(self):
        """返回迄今为止的最佳超参数。"""

        optimal_name = self.optimal_name

        return self._get_params_from_name(optimal_name)

    def clear(self):
        """清除所有先前结果和已保存的参数。"""
        shutil.rmtree(self.hyperparam_folder)
        os.makedirs(self.hyperparam_folder)
        self.results = pd.DataFrame()
        self.saved_params = pd.DataFrame()

    def _check_params(self, params):
        """检查参数映射是否正确定义。"""

        valid_fields = list(self.param_ranges.keys()) + list(self.fixed_params.keys())
        invalid_fields = [k for k in params if k not in valid_fields]
        missing_fields = [k for k in valid_fields if k not in params]

        if invalid_fields:
            raise ValueError("发现无效字段 {} - 有效字段为 {}".format(invalid_fields, valid_fields))
        if missing_fields:
            raise ValueError("发现缺失字段 {} - 有效字段为 {}".format(missing_fields, valid_fields))

    def _get_name(self, params):
        """为提供的参数集返回一个唯一的键。"""

        self._check_params(params)

        fields = list(params.keys())
        fields.sort()

        return "_".join([str(params[k]) for k in fields])

    def get_next_parameters(self, ranges_to_skip=None):
        """返回要优化的下一组参数。

        参数:
          ranges_to_skip: 显式定义一组要跳过的键。
        """
        if ranges_to_skip is None:
            ranges_to_skip = set(self.results.index)

        if not isinstance(self.param_ranges, dict):
            raise ValueError("仅适用于随机搜索！")

        param_range_keys = list(self.param_ranges.keys())
        param_range_keys.sort()

        def _get_next():
            """每次尝试返回下一个超参数集。"""

            parameters = {k: np.random.choice(self.param_ranges[k]) for k in param_range_keys}

            # 添加固定参数
            for k in self.fixed_params:
                parameters[k] = self.fixed_params[k]

            return parameters

        for _ in range(self._max_tries):
            parameters = _get_next()
            name = self._get_name(parameters)

            if name not in ranges_to_skip:
                return parameters

        raise ValueError("超出最大超参数搜索次数！！")

    def update_score(self, parameters, loss, model, info=""):
        """从上一次优化运行中更新结果。

        参数:
          parameters: 优化中使用的超参数。
          loss: 获得的验证损失。
          model: 如果需要，要序列化的模型。
          info: 要附加到结果的任何辅助信息。

        返回:
          一个布尔标志，指示该模型是否是迄今为止最好的模型。
        """

        if np.isnan(loss):
            loss = np.Inf

        if not os.path.isdir(self.hyperparam_folder):
            os.makedirs(self.hyperparam_folder)

        name = self._get_name(parameters)

        is_optimal = self.results.empty or loss < self.best_score

        # 保存第一个模型
        if is_optimal:
            # 在更新信息之前，先尝试保存
            if model is not None:
                print("找到最佳模型，正在更新")
                model.save(self.hyperparam_folder)
            self.best_score = loss
            self.optimal_name = name

        self.results[name] = pd.Series({"loss": loss, "info": info})
        self.saved_params[name] = pd.Series(parameters)

        self.results.to_csv(os.path.join(self.hyperparam_folder, "results.csv"))
        self.saved_params.to_csv(os.path.join(self.hyperparam_folder, "params.csv"))

        return is_optimal


class DistributedHyperparamOptManager(HyperparamOptManager):
    """管理跨多个 gpu 的分布式超参数优化。"""

    def __init__(
        self,
        param_ranges,
        fixed_params,
        root_model_folder,
        worker_number,
        search_iterations=1000,
        num_iterations_per_worker=5,
        clear_serialised_params=False,
    ):
        """实例化优化管理器。

        此超参数优化会预先生成 #search_iterations 个超参数组合，并在开始时将其序列化。在运行时，每个工作程序都会遍历自己的一组参数范围。预生成允许多个工作程序在不同的机器上并行运行，而不会导致参数重叠。

        参数:
          param_ranges: 用于随机搜索的离散超参数范围。
          fixed_params: 每个实验的固定模型参数。
          root_model_folder: 存储优化工件的文件夹。
          worker_number: 定义要测试的超参数集的工人索引。
          search_iterations: 最大随机搜索迭代次数。
          num_iterations_per_worker: 每个工人处理多少次迭代。
          clear_serialised_params: 是否重新生成超参数组合。
        """

        max_workers = int(np.ceil(search_iterations / num_iterations_per_worker))

        # 健全性检查
        if worker_number > max_workers:
            raise ValueError("工人编号 ({}) 不能大于总工人数量！".format(max_workers))
        if worker_number > search_iterations:
            raise ValueError(
                "工人编号 ({}) 不能大于最大搜索迭代次数 ({})！".format(
                    worker_number, search_iterations
                )
            )

        print("*** 正在为工人 {} 创建超参数管理器 ***".format(worker_number))

        hyperparam_folder = os.path.join(root_model_folder, str(worker_number))
        super().__init__(param_ranges, fixed_params, hyperparam_folder, override_w_fixed_params=True)

        serialised_ranges_folder = os.path.join(root_model_folder, "hyperparams")
        if clear_serialised_params:
            print("正在重新生成超参数列表")
            if os.path.exists(serialised_ranges_folder):
                shutil.rmtree(serialised_ranges_folder)

        utils.create_folder_if_not_exist(serialised_ranges_folder)

        self.serialised_ranges_path = os.path.join(serialised_ranges_folder, "ranges_{}.csv".format(search_iterations))
        self.hyperparam_folder = hyperparam_folder  # 覆盖
        self.worker_num = worker_number
        self.total_search_iterations = search_iterations
        self.num_iterations_per_worker = num_iterations_per_worker
        self.global_hyperparam_df = self.load_serialised_hyperparam_df()
        self.worker_search_queue = self._get_worker_search_queue()

    @property
    def optimisation_completed(self):
        """
        检查优化是否完成。
        """
        return False if self.worker_search_queue else True

    def get_next_parameters(self):
        """返回要优化的下一个超参数字典。"""
        param_name = self.worker_search_queue.pop()

        params = self.global_hyperparam_df.loc[param_name, :].to_dict()

        # 始终覆盖！
        for k in self.fixed_params:
            print("正在覆盖已保存的 {}: {}".format(k, self.fixed_params[k]))

            params[k] = self.fixed_params[k]

        return params

    def load_serialised_hyperparam_df(self):
        """从文件加载序列化的超参数范围。

        返回:
          包含超参数组合的数据帧。
        """
        print(
            "正在从 {} 加载 {} 次搜索迭代的参数".format(
                self.serialised_ranges_path, self.total_search_iterations
            )
        )

        if os.path.exists(self.serialised_ranges_folder):
            df = pd.read_csv(self.serialised_ranges_path, index_col=0)
        else:
            print("无法加载 - 正在重新生成搜索范围")
            df = self.update_serialised_hyperparam_df()

        return df

    def update_serialised_hyperparam_df(self):
        """重新生成超参数组合并保存到文件。

        返回:
          包含超参数组合的数据帧。
        """
        search_df = self._generate_full_hyperparam_df()

        print(
            "正在将 {} 次搜索迭代的参数序列化到 {}".format(
                self.total_search_iterations, self.serialised_ranges_path
            )
        )

        search_df.to_csv(self.serialised_ranges_path)

        return search_df

    def _generate_full_hyperparam_df(self):
        """生成实际的超参数组合。

        返回:
          包含超参数组合的数据帧。
        """

        np.random.seed(131)  # 为了超参数列表的可复现性

        name_list = []
        param_list = []
        for _ in range(self.total_search_iterations):
            params = super().get_next_parameters(name_list)

            name = self._get_name(params)

            name_list.append(name)
            param_list.append(params)

        full_search_df = pd.DataFrame(param_list, index=name_list)

        return full_search_df

    def clear(self):
        """清除超参数管理器的结果并重置。"""
        super().clear()
        self.worker_search_queue = self._get_worker_search_queue()

    def load_results(self):
        """从文件加载结果并将参数组合排队以进行尝试。

        返回:
          一个布尔值，指示是否成功加载结果。
        """
        success = super().load_results()

        if success:
            self.worker_search_queue = self._get_worker_search_queue()

        return success

    def _get_worker_search_queue(self):
        """为当前工人生成参数组合队列。

        返回:
          待处理的超参数组合队列。
        """
        global_df = self.assign_worker_numbers(self.global_hyperparam_df)
        worker_df = global_df[global_df["worker"] == self.worker_num]

        left_overs = [s for s in worker_df.index if s not in self.results.columns]

        return Deque(left_overs)

    def assign_worker_numbers(self, df):
        """用使用的工人索引更新参数组合。

        参数:
          df: 参数组合的数据帧。

        返回:
          带有工人编号的更新数据帧。
        """
        output = df.copy()

        n = self.total_search_iterations
        batch_size = self.num_iterations_per_worker

        max_worker_num = int(np.ceil(n / batch_size))

        worker_idx = np.concatenate([np.tile(i + 1, self.num_iterations_per_worker) for i in range(max_worker_num)])

        output["worker"] = worker_idx[: len(output)]

        return output
