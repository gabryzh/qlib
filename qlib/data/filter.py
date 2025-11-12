# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

from __future__ import print_function
from abc import abstractmethod

import re
import pandas as pd
import numpy as np
import abc

from .data import Cal, DatasetD


class BaseDFilter(abc.ABC):
    """动态金融工具过滤器抽象基类

    用户可以重写此类来构建自己的过滤器

    重写 __init__ 来输入过滤规则

    重写 filter_main 来使用规则过滤金融工具
    """

    def __init__(self):
        pass

    @staticmethod
    def from_config(config):
        """从配置字典构造一个实例。

        参数
        ----------
        config : dict
            配置参数字典。
        """
        raise NotImplementedError("BaseDFilter 的子类必须重新实现 `from_config` 方法")

    @abstractmethod
    def to_config(self):
        """将实例转换为配置字典。

        返回
        ----------
        dict
            返回配置参数字典。
        """
        raise NotImplementedError("BaseDFilter 的子类必须重新实现 `to_config` 方法")


class SeriesDFilter(BaseDFilter):
    """动态金融工具过滤器抽象类，用于过滤特定特征的序列

    过滤器应提供以下参数：

    - 过滤开始时间
    - 过滤结束时间
    - 过滤规则

    重写 __init__ 来为过滤序列分配特定规则。

    重写 _getFilterSeries 来使用规则过滤序列并获取 {inst => series} 的字典，或者重写 filter_main 来实现更高级的序列过滤规则
    """

    def __init__(self, fstart_time=None, fend_time=None, keep=False):
        """过滤器基类的初始化函数。
            根据 fstart_time 和 fend_time 指定的某个时期内的某个规则来过滤一组金融工具。

        参数
        ----------
        fstart_time: str
            过滤规则开始过滤金融工具的时间。
        fend_time: str
            过滤规则停止过滤金融工具的时间。
        keep: bool
            是否保留在过滤时间跨度内特征不存在的金融工具。
        """
        super(SeriesDFilter, self).__init__()
        self.filter_start_time = pd.Timestamp(fstart_time) if fstart_time else None
        self.filter_end_time = pd.Timestamp(fend_time) if fend_time else None
        self.keep = keep

    def _getTimeBound(self, instruments):
        """获取所有金融工具的时间边界。

        参数
        ----------
        instruments: dict
            金融工具字典，格式为 {instrument_name => list of timestamp tuple}。

        返回
        ----------
        pd.Timestamp, pd.Timestamp
            所有金融工具的下时间边界和上时间边界。
        """
        trange = Cal.calendar(freq=self.filter_freq)
        ubound, lbound = trange[0], trange[-1]
        for _, timestamp in instruments.items():
            if timestamp:
                lbound = timestamp[0][0] if timestamp[0][0] < lbound else lbound
                ubound = timestamp[-1][-1] if timestamp[-1][-1] > ubound else ubound
        return lbound, ubound

    def _toSeries(self, time_range, target_timestamp):
        """在时间范围内将目标时间戳转换为布尔值的 pandas 序列。
            使目标时间戳范围内的时间为 TRUE，其他为 FALSE。

        参数
        ----------
        time_range : D.calendar
            金融工具的时间范围。
        target_timestamp : list
            元组 (timestamp, timestamp) 的列表。

        返回
        ----------
        pd.Series
            一个金融工具的布尔值序列。
        """
        # 构造一个完整的 {date => bool} 字典
        timestamp_series = {timestamp: False for timestamp in time_range}
        # 转换为 pd.Series
        timestamp_series = pd.Series(timestamp_series)
        # 将 target_timestamp 内的日期填充为 TRUE
        for start, end in target_timestamp:
            timestamp_series[Cal.calendar(start_time=start, end_time=end, freq=self.filter_freq)] = True
        return timestamp_series

    def _filterSeries(self, timestamp_series, filter_series):
        """通过对两个序列进行逐元素 AND 运算来用过滤序列过滤时间戳序列。

        参数
        ----------
        timestamp_series : pd.Series
            指示存在时间的布尔值序列。
        filter_series : pd.Series
            指示过滤特征的布尔值序列。

        返回
        ----------
        pd.Series
            指示日期是否满足过滤条件并存在于目标时间戳中的布尔值序列。
        """
        fstart, fend = list(filter_series.keys())[0], list(filter_series.keys())[-1]
        filter_series = filter_series.astype("bool")  # 确保 filter_series 是布尔值
        timestamp_series[fstart:fend] = timestamp_series[fstart:fend] & filter_series
        return timestamp_series

    def _toTimestamp(self, timestamp_series):
        """将时间戳序列转换为元组 (timestamp, timestamp) 的列表，指示连续的 TRUE 范围。

        参数
        ----------
        timestamp_series: pd.Series
            过滤后的布尔值序列。

        返回
        ----------
        list
            元组 (timestamp, timestamp) 的列表。
        """
        # 根据时间戳对 timestamp_series 进行排序
        timestamp_series.sort_index()
        timestamp = []
        _lbool = None
        _ltime = None
        _cur_start = None
        for _ts, _bool in timestamp_series.items():
            # 当过滤序列没有布尔值时，可能会出现 NAN，我们只需将 NAN 更改为 False
            if pd.isna(_bool):
                _bool = False
            if _lbool is None:
                _cur_start = _ts
                _lbool = _bool
                _ltime = _ts
                continue
            if (_lbool, _bool) == (True, False):
                if _cur_start:
                    timestamp.append((_cur_start, _ltime))
            elif (_lbool, _bool) == (False, True):
                _cur_start = _ts
            _lbool = _bool
            _ltime = _ts
        if _lbool:
            timestamp.append((_cur_start, _ltime))
        return timestamp

    def __call__(self, instruments, start_time=None, end_time=None, freq="day"):
        """调用此过滤器以获取过滤后的金融工具列表"""
        self.filter_freq = freq
        return self.filter_main(instruments, start_time, end_time)

    @abstractmethod
    def _getFilterSeries(self, instruments, fstart, fend):
        """根据初始化期间分配的规则和输入的时间范围获取过滤序列。

        参数
        ----------
        instruments : dict
            要过滤的金融工具字典。
        fstart : pd.Timestamp
            过滤开始时间。
        fend : pd.Timestamp
            过滤结束时间。

        .. note:: fstart/fend 表示金融工具开始/结束时间与过滤开始/结束时间的交集。

        返回
        ----------
        pd.DataFrame
            一个 {pd.Timestamp => bool} 的序列。
        """
        raise NotImplementedError("SeriesDFilter 的子类必须重新实现 `getFilterSeries` 方法")

    def filter_main(self, instruments, start_time=None, end_time=None):
        """实现此方法来过滤金融工具。

        参数
        ----------
        instruments: dict
            要过滤的输入金融工具。
        start_time: str
            时间范围的开始。
        end_time: str
            时间范围的结束。

        返回
        ----------
        dict
            过滤后的金融工具，结构与输入金融工具相同。
        """
        lbound, ubound = self._getTimeBound(instruments)
        start_time = pd.Timestamp(start_time or lbound)
        end_time = pd.Timestamp(end_time or ubound)
        _instruments_filtered = {}
        _all_calendar = Cal.calendar(start_time=start_time, end_time=end_time, freq=self.filter_freq)
        _filter_calendar = Cal.calendar(
            start_time=self.filter_start_time and max(self.filter_start_time, _all_calendar[0]) or _all_calendar[0],
            end_time=self.filter_end_time and min(self.filter_end_time, _all_calendar[-1]) or _all_calendar[-1],
            freq=self.filter_freq,
        )
        _all_filter_series = self._getFilterSeries(instruments, _filter_calendar[0], _filter_calendar[-1])
        for inst, timestamp in instruments.items():
            # 构造一个完整的日期映射
            _timestamp_series = self._toSeries(_all_calendar, timestamp)
            # 获取过滤序列
            if inst in _all_filter_series:
                _filter_series = _all_filter_series[inst]
            else:
                if self.keep:
                    _filter_series = pd.Series({timestamp: True for timestamp in _filter_calendar})
                else:
                    _filter_series = pd.Series({timestamp: False for timestamp in _filter_calendar})
            # 计算过滤范围内的布尔值
            _timestamp_series = self._filterSeries(_timestamp_series, _filter_series)
            # 将映射重构成 (start_timestamp, end_timestamp) 格式
            _timestamp = self._toTimestamp(_timestamp_series)
            # 删除空时间戳
            if _timestamp:
                _instruments_filtered[inst] = _timestamp
        return _instruments_filtered


class NameDFilter(SeriesDFilter):
    """名称动态金融工具过滤器

    根据规范的名称格式过滤金融工具。

    需要一个名称规则正则表达式。
    """

    def __init__(self, name_rule_re, fstart_time=None, fend_time=None):
        """名称过滤器类的初始化函数

        参数
        ----------
        name_rule_re: str
            名称规则的正则表达式。
        """
        super(NameDFilter, self).__init__(fstart_time, fend_time)
        self.name_rule_re = name_rule_re

    def _getFilterSeries(self, instruments, fstart, fend):
        all_filter_series = {}
        filter_calendar = Cal.calendar(start_time=fstart, end_time=fend, freq=self.filter_freq)
        for inst, timestamp in instruments.items():
            if re.match(self.name_rule_re, inst):
                _filter_series = pd.Series({timestamp: True for timestamp in filter_calendar})
            else:
                _filter_series = pd.Series({timestamp: False for timestamp in filter_calendar})
            all_filter_series[inst] = _filter_series
        return all_filter_series

    @staticmethod
    def from_config(config):
        return NameDFilter(
            name_rule_re=config["name_rule_re"],
            fstart_time=config["filter_start_time"],
            fend_time=config["filter_end_time"],
        )

    def to_config(self):
        return {
            "filter_type": "NameDFilter",
            "name_rule_re": self.name_rule_re,
            "filter_start_time": str(self.filter_start_time) if self.filter_start_time else self.filter_start_time,
            "filter_end_time": str(self.filter_end_time) if self.filter_end_time else self.filter_end_time,
        }


class ExpressionDFilter(SeriesDFilter):
    """表达式动态金融工具过滤器

    根据特定表达式过滤金融工具。

    需要一个指示特定特征字段的表达式规则。

    示例
    ----------
    - *基本特征过滤器* : rule_expression = '$close/$open>5'
    - *横截面特征过滤器* : rule_expression = '$rank($close)<10'
    - *时间序列特征过滤器* : rule_expression = '$Ref($close, 3)>100'
    """

    def __init__(self, rule_expression, fstart_time=None, fend_time=None, keep=False):
        """表达式过滤器类的初始化函数

        参数
        ----------
        fstart_time: str
            从此时间开始过滤特征。
        fend_time: str
            到此时间结束过滤特征。
        rule_expression: str
            规则的输入表达式。
        """
        super(ExpressionDFilter, self).__init__(fstart_time, fend_time, keep=keep)
        self.rule_expression = rule_expression

    def _getFilterSeries(self, instruments, fstart, fend):
        # 不使用数据集缓存
        try:
            _features = DatasetD.dataset(
                instruments,
                [self.rule_expression],
                fstart,
                fend,
                freq=self.filter_freq,
                disk_cache=0,
            )
        except TypeError:
            # 使用 LocalDatasetProvider
            _features = DatasetD.dataset(instruments, [self.rule_expression], fstart, fend, freq=self.filter_freq)
        rule_expression_field_name = list(_features.keys())[0]
        all_filter_series = _features[rule_expression_field_name]
        return all_filter_series

    @staticmethod
    def from_config(config):
        return ExpressionDFilter(
            rule_expression=config["rule_expression"],
            fstart_time=config["filter_start_time"],
            fend_time=config["filter_end_time"],
            keep=config["keep"],
        )

    def to_config(self):
        return {
            "filter_type": "ExpressionDFilter",
            "rule_expression": self.rule_expression,
            "filter_start_time": str(self.filter_start_time) if self.filter_start_time else self.filter_start_time,
            "filter_end_time": str(self.filter_end_time) if self.filter_end_time else self.filter_end_time,
            "keep": self.keep,
        }
