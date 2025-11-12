import numpy as np
import pandas as pd
import importlib
from qlib.data.ops import ElemOperator, PairOperator
from qlib.config import C
from qlib.data.cache import H
from qlib.data.data import Cal
from qlib.contrib.ops.high_freq import get_calendar_day


class DayLast(ElemOperator):
    """DayLast 算子

    参数
    ----------
    feature : Expression
        特征实例

    返回
    ----------
    feature:
        一个序列，其中每个值等于其所在天的最后一个值
    """

    def _load_internal(self, instrument, start_index, end_index, freq):
        _calendar = get_calendar_day(freq=freq)
        series = self.feature.load(instrument, start_index, end_index, freq)
        return series.groupby(_calendar[series.index], group_keys=False).transform("last")


class FFillNan(ElemOperator):
    """FFillNan 算子

    参数
    ----------
    feature : Expression
        特征实例

    返回
    ----------
    feature:
        一个向前填充 nan 的特征
    """

    def _load_internal(self, instrument, start_index, end_index, freq):
        series = self.feature.load(instrument, start_index, end_index, freq)
        return series.ffill()


class BFillNan(ElemOperator):
    """BFillNan 算子

    参数
    ----------
    feature : Expression
        特征实例

    返回
    ----------
    feature:
        一个向后填充 nan 的特征
    """

    def _load_internal(self, instrument, start_index, end_index, freq):
        series = self.feature.load(instrument, start_index, end_index, freq)
        return series.bfill()


class Date(ElemOperator):
    """Date 算子

    参数
    ----------
    feature : Expression
        特征实例

    返回
    ----------
    feature:
        一个序列，其中每个值是对应于 feature.index 的日期
    """

    def _load_internal(self, instrument, start_index, end_index, freq):
        _calendar = get_calendar_day(freq=freq)
        series = self.feature.load(instrument, start_index, end_index, freq)
        return pd.Series(_calendar[series.index], index=series.index)


class Select(PairOperator):
    """Select 算子

    参数
    ----------
    feature_left : Expression
        特征实例，选择条件
    feature_right : Expression
        特征实例，选择值

    返回
    ----------
    feature:
        满足条件(feature_left)的值(feature_right)

    """

    def _load_internal(self, instrument, start_index, end_index, freq):
        series_condition = self.feature_left.load(instrument, start_index, end_index, freq)
        series_feature = self.feature_right.load(instrument, start_index, end_index, freq)
        return series_feature.loc[series_condition]


class IsNull(ElemOperator):
    """IsNull 算子

    参数
    ----------
    feature : Expression
        特征实例

    返回
    ----------
    feature:
        一个指示特征是否为 nan 的序列
    """

    def _load_internal(self, instrument, start_index, end_index, freq):
        series = self.feature.load(instrument, start_index, end_index, freq)
        return series.isnull()


class Cut(ElemOperator):
    """Cut 算子

    参数
    ----------
    feature : Expression
        特征实例
    l : int
        l > 0, 删除特征的前 l 个元素 (默认为 None, 表示 0)
    r : int
        r < 0, 删除特征的后 -r 个元素 (默认为 None, 表示 0)
    返回
    ----------
    feature:
        一个从特征中删除了前 l 个和后 -r 个元素的序列。
        注意：它是从原始数据中删除，而不是从切片数据中删除
    """

    def __init__(self, feature, l=None, r=None):
        self.l = l
        self.r = r
        if (self.l is not None and self.l <= 0) or (self.r is not None and self.r >= 0):
            raise ValueError("Cut 算子的 l 应 > 0 且 r 应 < 0")

        super(Cut, self).__init__(feature)

    def _load_internal(self, instrument, start_index, end_index, freq):
        series = self.feature.load(instrument, start_index, end_index, freq)
        return series.iloc[self.l : self.r]

    def get_extended_window_size(self):
        """
        获取扩展窗口大小。
        """
        ll = 0 if self.l is None else self.l
        rr = 0 if self.r is None else abs(self.r)
        lft_etd, rght_etd = self.feature.get_extended_window_size()
        lft_etd = lft_etd + ll
        rght_etd = rght_etd + rr
        return lft_etd, rght_etd
