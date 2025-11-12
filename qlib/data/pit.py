# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.
"""
Qlib 遵循以下逻辑来支持即时（Point-in-Time）数据库

对于每只股票，其数据格式为 <观察时间, 特征>。表达式引擎支持对这种格式的数据进行计算。

要在特定的观察时间 t 计算特征值 f_t，将使用 <时期时间, 特征> 格式的数据。
例如，在 20190719（观察时间）计算过去 4 个季度（时期时间）的平均收益。

<时期时间, 特征> 和 <观察时间, 特征> 数据的计算都依赖于表达式引擎。它包括两个阶段：
1) 在每个观察时间 t 计算 <时期时间, 特征>，并将其折叠成一个点（就像一个普通的特征）。
2) 连接所有折叠后的数据，我们将得到 <观察时间, 特征> 格式的数据。
Qlib 将使用 `P` 运算符来执行折叠操作。
"""
import numpy as np
import pandas as pd
from qlib.data.ops import ElemOperator
from qlib.log import get_module_logger
from .data import Cal


class P(ElemOperator):
    """即时（Point-in-Time）运算符

    `P` 运算符用于将 <时期时间, 特征> 格式的数据转换为 <观察时间, 特征> 格式。
    它在每个观察时间点对时期数据进行计算，并将结果折叠成一个单一的值。
    """
    def _load_internal(self, instrument, start_index, end_index, freq):
        _calendar = Cal.calendar(freq=freq)
        resample_data = np.empty(end_index - start_index + 1, dtype="float32")

        for cur_index in range(start_index, end_index + 1):
            cur_time = _calendar[cur_index]
            # 为了准确加载表达式，需要更多的历史数据
            start_ws, end_ws = self.feature.get_extended_window_size()
            if end_ws > 0:
                raise ValueError(
                    "即时数据库不支持引用未来时期（例如，不支持像 `Ref('$$roewa_q', -1)` 这样的表达式"
                )

            # 计算出的值将始终是最后一个元素，因此 end_offset 为零。
            try:
                s = self._load_feature(instrument, -start_ws, 0, cur_time)
                resample_data[cur_index - start_index] = s.iloc[-1] if len(s) > 0 else np.nan
            except FileNotFoundError:
                get_module_logger("base").warning(f"警告：未找到 {str(self)} 的时期数据")
                return pd.Series(dtype="float32", name=str(self))

        resample_series = pd.Series(
            resample_data, index=pd.RangeIndex(start_index, end_index + 1), dtype="float32", name=str(self)
        )
        return resample_series

    def _load_feature(self, instrument, start_index, end_index, cur_time):
        return self.feature.load(instrument, start_index, end_index, cur_time)

    def get_longest_back_rolling(self):
        # 时期数据将折叠为普通特征。因此没有扩展和回溯
        return 0

    def get_extended_window_size(self):
        # 时期数据将折叠为普通特征。因此没有扩展和回溯
        return 0, 0


class PRef(P):
    """带特定时期引用的即时（Point-in-Time）运算符

    `PRef` 运算符用于在特定时期查询即时数据。
    """
    def __init__(self, feature, period):
        super().__init__(feature)
        self.period = period

    def __str__(self):
        return f"{super().__str__()}[{self.period}]"

    def _load_feature(self, instrument, start_index, end_index, cur_time):
        return self.feature.load(instrument, start_index, end_index, cur_time, self.period)
