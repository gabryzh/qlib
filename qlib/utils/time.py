# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.
"""
此脚本中编译了与时间相关的实用程序
"""
import bisect
from datetime import datetime, time, date, timedelta
from typing import List, Optional, Tuple, Union
import functools
import re

import pandas as pd

from qlib.config import C
from qlib.constant import REG_CN, REG_TW, REG_US


CN_TIME = [
    datetime.strptime("9:30", "%H:%M"),
    datetime.strptime("11:30", "%H:%M"),
    datetime.strptime("13:00", "%H:%M"),
    datetime.strptime("15:00", "%H:%M"),
]
US_TIME = [datetime.strptime("9:30", "%H:%M"), datetime.strptime("16:00", "%H:%M")]
TW_TIME = [
    datetime.strptime("9:00", "%H:%M"),
    datetime.strptime("13:30", "%H:%M"),
]


@functools.lru_cache(maxsize=240)
def get_min_cal(shift: int = 0, region: str = REG_CN) -> List[time]:
    """
    获取一天中的分钟级别日历

    参数
    ----------
    shift : int
        偏移方向类似于pandas的shift。
        series.shift(1)会将第i个位置的值替换为第i-1个位置的值
    region: str
        区域，例如 "cn", "us"

    返回
    -------
    List[time]:
        分钟级别日历

    """
    cal = []

    if region == REG_CN:
        for ts in list(
            pd.date_range(CN_TIME[0], CN_TIME[1] - timedelta(minutes=1), freq="1min") - pd.Timedelta(minutes=shift)
        ) + list(
            pd.date_range(CN_TIME[2], CN_TIME[3] - timedelta(minutes=1), freq="1min") - pd.Timedelta(minutes=shift)
        ):
            cal.append(ts.time())
    elif region == REG_TW:
        for ts in list(
            pd.date_range(TW_TIME[0], TW_TIME[1] - timedelta(minutes=1), freq="1min") - pd.Timedelta(minutes=shift)
        ):
            cal.append(ts.time())
    elif region == REG_US:
        for ts in list(
            pd.date_range(US_TIME[0], US_TIME[1] - timedelta(minutes=1), freq="1min") - pd.Timedelta(minutes=shift)
        ):
            cal.append(ts.time())
    else:
        raise ValueError(f"{region} is not supported")
    return cal


def is_single_value(start_time, end_time, freq, region: str = REG_CN):
    """股市是否只有一条数据。

    参数
    ----------
    start_time : Union[pd.Timestamp, str]
        数据的平仓开始时间。
    end_time : Union[pd.Timestamp, str]
        数据的平仓结束时间。
    freq :
        频率。
    region: str
        区域，例如 "cn", "us"。
    返回
    -------
    bool
        True 表示获取一条数据。
    """
    if region == REG_CN:
        if end_time - start_time < freq:
            return True
        if start_time.hour == 11 and start_time.minute == 29 and start_time.second == 0:
            return True
        if start_time.hour == 14 and start_time.minute == 59 and start_time.second == 0:
            return True
        return False
    elif region == REG_TW:
        if end_time - start_time < freq:
            return True
        if start_time.hour == 13 and start_time.minute >= 25 and start_time.second == 0:
            return True
        return False
    elif region == REG_US:
        if end_time - start_time < freq:
            return True
        if start_time.hour == 15 and start_time.minute == 59 and start_time.second == 0:
            return True
        return False
    else:
        raise NotImplementedError(f"please implement the is_single_value func for {region}")


class Freq:
    NORM_FREQ_MONTH = "month"
    NORM_FREQ_WEEK = "week"
    NORM_FREQ_DAY = "day"
    NORM_FREQ_MINUTE = "min"  # using min instead of minute for align with Qlib's data filename
    SUPPORT_CAL_LIST = [NORM_FREQ_MINUTE, NORM_FREQ_DAY]  # FIXME: 此列表应来自数据

    def __init__(self, freq: Union[str, "Freq"]) -> None:
        if isinstance(freq, str):
            self.count, self.base = self.parse(freq)
        elif isinstance(freq, Freq):
            self.count, self.base = freq.count, freq.base
        else:
            raise NotImplementedError(f"不支持此类型的输入")

    def __eq__(self, freq):
        freq = Freq(freq)
        return freq.count == self.count and freq.base == self.base

    def __str__(self):
        # trying to align to the filename of Qlib: day, 30min, 5min, 1min...
        return f"{self.count if self.count != 1 or self.base != 'day' else ''}{self.base}"

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}({str(self)})"

    @staticmethod
    def parse(freq: str) -> Tuple[int, str]:
        """
        将频率解析为统一格式

        参数
        ----------
        freq : str
            原始频率，支持的频率应匹配正则表达式'^([0-9]*)(month|mon|week|w|day|d|minute|min)$'

        返回
        -------
        freq: Tuple[int, str]
            统一的频率，包括频率计数和统一的频率单位。频率单位应为'[month|week|day|minute]'。
                示例:

                .. code-block::

                    print(Freq.parse("day"))
                    (1, "day" )
                    print(Freq.parse("2mon"))
                    (2, "month")
                    print(Freq.parse("10w"))
                    (10, "week")

        """
        freq = freq.lower()
        match_obj = re.match("^([0-9]*)(month|mon|week|w|day|d|minute|min)$", freq)
        if match_obj is None:
            raise ValueError(
                "不支持的频率格式，频率应类似于(n)month/mon, (n)week/w, (n)day/d, (n)minute/min"
            )
        _count = int(match_obj.group(1)) if match_obj.group(1) else 1
        _freq = match_obj.group(2)
        _freq_format_dict = {
            "month": Freq.NORM_FREQ_MONTH,
            "mon": Freq.NORM_FREQ_MONTH,
            "week": Freq.NORM_FREQ_WEEK,
            "w": Freq.NORM_FREQ_WEEK,
            "day": Freq.NORM_FREQ_DAY,
            "d": Freq.NORM_FREQ_DAY,
            "minute": Freq.NORM_FREQ_MINUTE,
            "min": Freq.NORM_FREQ_MINUTE,
        }
        return _count, _freq_format_dict[_freq]

    @staticmethod
    def get_timedelta(n: int, freq: str) -> pd.Timedelta:
        """
        获取pd.Timedelta对象

        参数
        ----------
        n : int
        freq : str
            通常，它们是Freq.parse的返回值

        返回
        -------
        pd.Timedelta:
            时间差对象
        """
        return pd.Timedelta(f"{n}{freq}")

    @staticmethod
    def get_min_delta(left_frq: str, right_freq: str):
        """计算频率增量

        参数
        ----------
        left_frq: str
            左侧频率
        right_freq: str
            右侧频率

        返回
        -------
        int
            分钟级别的频率增量
        """
        minutes_map = {
            Freq.NORM_FREQ_MINUTE: 1,
            Freq.NORM_FREQ_DAY: 60 * 24,
            Freq.NORM_FREQ_WEEK: 7 * 60 * 24,
            Freq.NORM_FREQ_MONTH: 30 * 7 * 60 * 24,
        }
        left_freq = Freq(left_frq)
        left_minutes = left_freq.count * minutes_map[left_freq.base]
        right_freq = Freq(right_freq)
        right_minutes = right_freq.count * minutes_map[right_freq.base]
        return left_minutes - right_minutes

    @staticmethod
    def get_recent_freq(base_freq: Union[str, "Freq"], freq_list: List[Union[str, "Freq"]]) -> Optional["Freq"]:
        """从freq_list中获取最接近base_freq的频率

        参数
        ----------
        base_freq
            基础频率
        freq_list
            频率列表

        返回
        -------
        如果找到最近的频率
            Freq
        否则:
            None
        """
        base_freq = Freq(base_freq)
        # use the nearest freq greater than 0
        min_freq = None
        for _freq in freq_list:
            _min_delta = Freq.get_min_delta(base_freq, _freq)
            if _min_delta < 0:
                continue
            if min_freq is None:
                min_freq = (_min_delta, str(_freq))
                continue
            min_freq = min_freq if min_freq[0] <= _min_delta else (_min_delta, _freq)
        return min_freq[1] if min_freq else None


def time_to_day_index(time_obj: Union[str, datetime], region: str = REG_CN):
    """将时间对象转换为天索引"""
    if isinstance(time_obj, str):
        time_obj = datetime.strptime(time_obj, "%H:%M")

    if region == REG_CN:
        if CN_TIME[0] <= time_obj < CN_TIME[1]:
            return int((time_obj - CN_TIME[0]).total_seconds() / 60)
        elif CN_TIME[2] <= time_obj < CN_TIME[3]:
            return int((time_obj - CN_TIME[2]).total_seconds() / 60) + 120
        else:
            raise ValueError(f"{time_obj} 不是 {region} 股市的开盘时间")
    elif region == REG_US:
        if US_TIME[0] <= time_obj < US_TIME[1]:
            return int((time_obj - US_TIME[0]).total_seconds() / 60)
        else:
            raise ValueError(f"{time_obj} 不是 {region} 股市的开盘时间")
    elif region == REG_TW:
        if TW_TIME[0] <= time_obj < TW_TIME[1]:
            return int((time_obj - TW_TIME[0]).total_seconds() / 60)
        else:
            raise ValueError(f"{time_obj} 不是 {region} 股市的开盘时间")
    else:
        raise ValueError(f"不支持区域 {region}")


def get_day_min_idx_range(start: str, end: str, freq: str, region: str) -> Tuple[int, int]:
    """
    在给定固定频率的情况下，获取一天中某个时间范围（左右都关闭）的分钟柱索引
    参数
    ----------
    start : str
        例如 "9:30"
    end : str
        例如 "14:30"
    freq : str
        "1min"

    返回
    -------
    Tuple[int, int]:
        日历中开始和结束的索引。左右都**关闭**
    """
    start = pd.Timestamp(start).time()
    end = pd.Timestamp(end).time()
    freq = Freq(freq)
    in_day_cal = get_min_cal(region=region)[:: freq.count]
    left_idx = bisect.bisect_left(in_day_cal, start)
    right_idx = bisect.bisect_right(in_day_cal, end) - 1
    return left_idx, right_idx


def concat_date_time(date_obj: date, time_obj: time) -> pd.Timestamp:
    return pd.Timestamp(
        datetime(
            date_obj.year,
            month=date_obj.month,
            day=date_obj.day,
            hour=time_obj.hour,
            minute=time_obj.minute,
            second=time_obj.second,
            microsecond=time_obj.microsecond,
        )
    )


def cal_sam_minute(x: pd.Timestamp, sam_minutes: int, region: str = REG_CN) -> pd.Timestamp:
    """
    将分钟级别数据与下采样日历对齐

    例如，在5分钟级别将10:38对齐到10:35（在10分钟级别对齐到10:30）

    参数
    ----------
    x : pd.Timestamp
        要对齐的日期时间
    sam_minutes : int
        对齐到 `sam_minutes` 分钟级别日历
    region: str
        区域，例如 "cn", "us"

    返回
    -------
    pd.Timestamp:
        对齐后的日期时间
    """
    cal = get_min_cal(C.min_data_shift, region)[::sam_minutes]
    idx = bisect.bisect_right(cal, x.time()) - 1
    _date, new_time = x.date(), cal[idx]
    return concat_date_time(_date, new_time)


def epsilon_change(date_time: pd.Timestamp, direction: str = "backward") -> pd.Timestamp:
    """
    按极小量改变时间。


    参数
    ----------
    date_time : pd.Timestamp
        原始时间
    direction : str
        时间变化的方向
        - "backward" 表示回到过去
        - "forward" 表示走向未来

    返回
    -------
    pd.Timestamp:
        偏移后的时间
    """
    if direction == "backward":
        return date_time - pd.Timedelta(seconds=1)
    elif direction == "forward":
        return date_time + pd.Timedelta(seconds=1)
    else:
        raise ValueError("Wrong input")


if __name__ == "__main__":
    print(get_day_min_idx_range("8:30", "14:59", "10min", REG_CN))
