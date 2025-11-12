# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.


from __future__ import division
from __future__ import print_function

import abc
import pandas as pd
from ..log import get_module_logger


class Expression(abc.ABC):
    """表达式基类

    表达式旨在处理以下格式的数据计算：
    每个金融工具（例如股票）的数据包含两个维度：

    - 特征（feature）
    - 时间（time）：可以是观察时间或时期时间。

        - 时期时间（period time）是为切点（Point-in-time）数据库设计的。例如，时期时间可能是 2014Q4，
          它的值可以被多次观察（由于修正，不同时间观察到的值可能不同）。
    """

    def __str__(self):
        return type(self).__name__

    def __repr__(self):
        return str(self)

    def __gt__(self, other):
        from .ops import Gt  # pylint: disable=C0415

        return Gt(self, other)

    def __ge__(self, other):
        from .ops import Ge  # pylint: disable=C0415

        return Ge(self, other)

    def __lt__(self, other):
        from .ops import Lt  # pylint: disable=C0415

        return Lt(self, other)

    def __le__(self, other):
        from .ops import Le  # pylint: disable=C0415

        return Le(self, other)

    def __eq__(self, other):
        from .ops import Eq  # pylint: disable=C0415

        return Eq(self, other)

    def __ne__(self, other):
        from .ops import Ne  # pylint: disable=C0415

        return Ne(self, other)

    def __add__(self, other):
        from .ops import Add  # pylint: disable=C0415

        return Add(self, other)

    def __radd__(self, other):
        from .ops import Add  # pylint: disable=C0415

        return Add(other, self)

    def __sub__(self, other):
        from .ops import Sub  # pylint: disable=C0415

        return Sub(self, other)

    def __rsub__(self, other):
        from .ops import Sub  # pylint: disable=C0415

        return Sub(other, self)

    def __mul__(self, other):
        from .ops import Mul  # pylint: disable=C0415

        return Mul(self, other)

    def __rmul__(self, other):
        from .ops import Mul  # pylint: disable=C0415

        return Mul(self, other)

    def __div__(self, other):
        from .ops import Div  # pylint: disable=C0415

        return Div(self, other)

    def __rdiv__(self, other):
        from .ops import Div  # pylint: disable=C0415

        return Div(other, self)

    def __truediv__(self, other):
        from .ops import Div  # pylint: disable=C0415

        return Div(self, other)

    def __rtruediv__(self, other):
        from .ops import Div  # pylint: disable=C0415

        return Div(other, self)

    def __pow__(self, other):
        from .ops import Power  # pylint: disable=C0415

        return Power(self, other)

    def __rpow__(self, other):
        from .ops import Power  # pylint: disable=C0415

        return Power(other, self)

    def __and__(self, other):
        from .ops import And  # pylint: disable=C0415

        return And(self, other)

    def __rand__(self, other):
        from .ops import And  # pylint: disable=C0415

        return And(other, self)

    def __or__(self, other):
        from .ops import Or  # pylint: disable=C0415

        return Or(self, other)

    def __ror__(self, other):
        from .ops import Or  # pylint: disable=C0415

        return Or(other, self)

    def load(self, instrument, start_index, end_index, *args):
        """加载特征

        此函数负责基于表达式引擎加载特征/表达式。

        具体实现分为两部分：
        1) 缓存数据，处理错误。
           - 这部分由所有表达式共享，并在 `Expression` 类中实现。
        2) 基于特定表达式处理和计算数据。
           - 这部分在每个表达式中都不同，并在各自的表达式类中实现。

        表达式引擎由不同的数据共享。
        不同的数据将为 `args` 提供不同的额外信息。

        参数
        ----------
        instrument : str
            金融工具代码（例如股票代码）。
        start_index : str
            特征的开始索引 [在日历中]。
        end_index : str
            特征的结束索引 [在日历中]。

        *args 可能包含以下信息：
        1) 如果在基本表达式引擎数据中使用，它包含以下参数：
            freq: str
                特征的频率。

        2) 如果在切点（PIT）数据中使用，它包含以下参数：
            cur_pit:
                为切点数据设计。
            period: int
                用于查询特定时期。
                在 Qlib 中，时期用整数表示（例如，202001 可能代表 2020 年第一季度）。

        返回
        -------
        pd.Series
            特征序列：序列的索引是日历索引。
        """
        from .cache import H  # pylint: disable=C0415

        # 缓存
        cache_key = str(self), instrument, start_index, end_index, *args
        if cache_key in H["f"]:
            return H["f"][cache_key]
        if start_index is not None and end_index is not None and start_index > end_index:
            raise ValueError("Invalid index range: {} {}".format(start_index, end_index))
        try:
            series = self._load_internal(instrument, start_index, end_index, *args)
        except Exception as e:
            get_module_logger("data").debug(
                f"Loading data error: instrument={instrument}, expression={str(self)}, "
                f"start_index={start_index}, end_index={end_index}, args={args}. "
                f"error info: {str(e)}"
            )
            raise
        series.name = str(self)
        H["f"][cache_key] = series
        return series

    @abc.abstractmethod
    def _load_internal(self, instrument, start_index, end_index, *args) -> pd.Series:
        """
        _load_internal 是一个用于加载数据的私有方法。
        它由 `load` 方法调用。
        子类必须实现此方法。
        """
        raise NotImplementedError("This function must be implemented in your newly defined feature")

    @abc.abstractmethod
    def get_longest_back_rolling(self):
        """获取特征访问过的历史数据的最长回溯长度

        这旨在首先获取计算特定范围内特征所需的数据范围。
        然而，像 `Ref(Ref($close, -1), 1)` 这样的情况无法正确处理。

        因此，这仅用于检测所需的历史数据长度。
        """
        # TODO: forward operator like Ref($close, -1) is not supported yet.
        raise NotImplementedError("This function must be implemented in your newly defined feature")

    @abc.abstractmethod
    def get_extended_window_size(self):
        """获取扩展窗口大小

        为了在范围 [start_index, end_index] 内计算此运算符，
        我们必须在范围 [start_index - lft_etd, end_index + rght_etd] 内获取 *叶节点特征*。

        返回
        ----------
        (int, int)
            lft_etd, rght_etd
        """
        raise NotImplementedError("This function must be implemented in your newly defined feature")


class Feature(Expression):
    """静态表达式

    这种特征将从数据提供程序加载数据。
    """

    def __init__(self, name=None):
        if name:
            self._name = name
        else:
            self._name = type(self).__name__

    def __str__(self):
        return "$" + self._name

    def _load_internal(self, instrument, start_index, end_index, freq):
        # 加载
        from .data import FeatureD  # pylint: disable=C0415

        return FeatureD.feature(instrument, str(self), start_index, end_index, freq)

    def get_longest_back_rolling(self):
        return 0

    def get_extended_window_size(self):
        return 0, 0


class PFeature(Feature):
    """切点（Point-in-time）特征"""
    def __str__(self):
        return "$$" + self._name

    def _load_internal(self, instrument, start_index, end_index, cur_time, period=None):
        from .data import PITD  # pylint: disable=C0415

        return PITD.period_feature(instrument, str(self), start_index, end_index, cur_time, period)


class ExpressionOps(Expression):
    """运算符表达式

    这种特征将使用运算符动态构建特征。
    """