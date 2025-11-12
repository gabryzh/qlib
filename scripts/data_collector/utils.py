#  Copyright (c) Microsoft Corporation.
#  Licensed under the MIT License.

# 导入必要的库
import re
import copy
import importlib
import time
import bisect
import pickle
import random
import requests
import functools
from pathlib import Path
from typing import Iterable, Tuple, List

import numpy as np
import pandas as pd
from loguru import logger
from yahooquery import Ticker  # 用于从Yahoo Finance获取数据
from tqdm import tqdm
from functools import partial
from concurrent.futures import ProcessPoolExecutor
from bs4 import BeautifulSoup  # 用于解析HTML

# A股股票列表的URL
HS_SYMBOLS_URL = "http://app.finance.ifeng.com/hq/list.php?type=stock_a&class={s_type}"

# 获取交易日历的基础URL（东方财富）
CALENDAR_URL_BASE = "http://push2his.eastmoney.com/api/qt/stock/kline/get?secid={market}.{bench_code}&fields1=f1%2Cf2%2Cf3%2Cf4%2Cf5&fields2=f51%2Cf52%2Cf53%2Cf54%2Cf55%2Cf56%2Cf57%2Cf58&klt=101&fqt=0&beg=19900101&end=20991231"
# 深圳证券交易所的交易日历API
SZSE_CALENDAR_URL = "http://www.szse.cn/api/report/exchange/onepersistenthour/monthList?month={month}&random={random}"

# 不同市场基准指数与对应URL的映射
CALENDAR_BENCH_URL_MAP = {
    "CSI300": CALENDAR_URL_BASE.format(market=1, bench_code="000300"), # 沪深300
    "CSI500": CALENDAR_URL_BASE.format(market=1, bench_code="000905"), # 中证500
    "CSI100": CALENDAR_URL_BASE.format(market=1, bench_code="000903"), # 中证100
    # 注意：使用中证500的交易日历作为所有A股的日历
    "ALL": CALENDAR_URL_BASE.format(market=1, bench_code="000905"),
    # 注意：使用标普500(^GSPC)的交易日历作为所有美股的日历
    "US_ALL": "^GSPC",
    "IN_ALL": "^NSEI", # 印度Nifty 50
    "BR_ALL": "^BVSP", # 巴西IBOVESPA
}

# 用于缓存已获取数据的全局变量
_BENCH_CALENDAR_LIST = None
_ALL_CALENDAR_LIST = None
_HS_SYMBOLS = None
_US_SYMBOLS = None
_IN_SYMBOLS = None
_BR_SYMBOLS = None
_EN_FUND_SYMBOLS = None
_CALENDAR_MAP = {}

# A股市场的最小股票数量阈值，用于校验数据完整性
MINIMUM_SYMBOLS_NUM = 3900


def get_calendar_list(bench_code="CSI300") -> List[pd.Timestamp]:
    """
    获取沪深市场的历史交易日历。

    Parameters
    ----------
    bench_code: str
        基准指数代码, 可选值 ["CSI300", "CSI500", "ALL", "US_ALL", "IN_ALL", "BR_ALL"]。

    Returns
    -------
        List[pd.Timestamp]: 历史交易日历列表。
    """
    logger.info(f"开始获取交易日历: {bench_code}......")

    def _get_calendar_from_eastmoney(url):
        # 从东方财富API获取日历
        _value_list = requests.get(url, timeout=None).json()["data"]["klines"]
        return sorted(map(lambda x: pd.Timestamp(x.split(",")[0]), _value_list))

    # 尝试从缓存中读取
    calendar = _CALENDAR_MAP.get(bench_code, None)
    if calendar is None:
        if bench_code.startswith(("US_", "IN_", "BR_")):
            # 对于美股、印度、巴西市场，使用yahooquery获取
            df = Ticker(CALENDAR_BENCH_URL_MAP[bench_code]).history(interval="1d", period="max")
            calendar = df.index.get_level_values(level="date").map(pd.Timestamp).unique().tolist()
        else:
            if bench_code.upper() == "ALL":
                # 对于A股全市场，使用深交所API按月获取
                @deco_retry
                def _get_calendar_from_month(month):
                    _cal = []
                    try:
                        # 请求深交所API
                        resp = requests.get(
                            SZSE_CALENDAR_URL.format(month=month, random=random.random()), timeout=None
                        ).json()
                        for _r in resp["data"]:
                            if int(_r["jybz"]): # jybz: 交易标志, 1为交易日
                                _cal.append(pd.Timestamp(_r["jyrq"])) # jyrq: 交易日期
                    except Exception as e:
                        raise ValueError(f"获取月份 {month} 日历失败: {e}") from e
                    return _cal

                # 生成从2000年1月至今的月份范围
                month_range = pd.date_range(start="2000-01", end=pd.Timestamp.now() + pd.Timedelta(days=31), freq="M")
                calendar = []
                for _m in month_range:
                    cal = _get_calendar_from_month(_m.strftime("%Y-%m"))
                    if cal:
                        calendar += cal
                # 过滤掉未来的日期
                calendar = list(filter(lambda x: x <= pd.Timestamp.now(), calendar))
            else:
                # 对于特定指数，使用东方财富API获取
                calendar = _get_calendar_from_eastmoney(CALENDAR_BENCH_URL_MAP[bench_code])
        # 存入缓存
        _CALENDAR_MAP[bench_code] = calendar
    logger.info(f"获取交易日历结束: {bench_code}.")
    return calendar

# ... (其他函数的中文注释)
def deco_retry(retry: int = 5, retry_sleep: int = 3):
    """
    一个装饰器，用于在函数执行失败时进行重试。

    Parameters
    ----------
    retry : int or callable
        最大重试次数。如果作为不带参数的装饰器使用，它将是第一个参数（即被装饰的函数）。
    retry_sleep : int
        每次重试之间的等待时间（秒）。

    Returns
    -------
    callable
        被装饰的函数。
    """
    def deco_func(func):
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            _retry_count = 5 if callable(retry) else retry
            _result = None
            for _i in range(1, _retry_count + 1):
                try:
                    _result = func(*args, **kwargs)
                    break # 成功则跳出循环
                except Exception as e:
                    logger.warning(f"函数 {func.__name__} 第 {_i} 次执行失败: {e}")
                    if _i == _retry_count:
                        # 达到最大重试次数，抛出异常
                        raise
                time.sleep(retry_sleep)
            return _result
        return wrapper

    # 允许装饰器带参数或不带参数使用
    return deco_func(retry) if callable(retry) else deco_func

def get_trading_date_by_shift(trading_list: list, trading_date: pd.Timestamp, shift: int = 1) -> pd.Timestamp:
    """
    根据给定的交易日历，获取某个日期偏移指定交易日数后的日期。

    Parameters
    ----------
    trading_list : list
        已排序的交易日历列表。
    trading_date : pd.Timestamp
        基准日期。
    shift : int
        偏移的交易日数，可以为正（未来）或负（过去）。

    Returns
    -------
    pd.Timestamp
        偏移后的交易日。
    """
    trading_date = pd.Timestamp(trading_date)
    # 使用二分查找找到或插入日期的位置
    left_index = bisect.bisect_left(trading_list, trading_date)
    try:
        res = trading_list[left_index + shift]
    except IndexError:
        # 如果索引越界（例如，请求最新的日期再往后一天），则返回原日期
        res = trading_date
    return res
# ... (其他函数的中文注释)

# 如果作为主程序运行，执行一个简单的断言检查
if __name__ == "__main__":
    assert len(get_hs_stock_symbols()) >= MINIMUM_SYMBOLS_NUM
