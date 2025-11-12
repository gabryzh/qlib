# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.


# 导入必要的库
import sys
import copy
import fire
import numpy as np
import pandas as pd
import baostock as bs  # baostock库，用于获取A股数据
from tqdm import tqdm
from pathlib import Path
from loguru import logger
from typing import Iterable, List

import qlib
from qlib.data import D

# 将上两级目录添加到系统路径，以便导入base和utils模块
CUR_DIR = Path(__file__).resolve().parent
sys.path.append(str(CUR_DIR.parent.parent))

from data_collector.base import BaseCollector, BaseNormalize, BaseRun
from data_collector.utils import generate_minutes_calendar_from_daily, calc_adjusted_price


class BaostockCollectorHS3005min(BaseCollector):
    """
    用于从Baostock收集沪深300成分股5分钟K线数据的收集器。
    """

    def __init__(
        self,
        save_dir: [str, Path],
        start=None,
        end=None,
        interval="5min",
        max_workers=4,
        max_collector_count=2,
        delay=0,
        check_data_length: int = None,
        limit_nums: int = None,
    ):
        """
        初始化。

        Parameters
        ----------
        save_dir: str
            数据保存目录。
        start: str
            开始日期时间。
        end: str
            结束日期时间。
        interval: str
            数据频率, 此处固定为 "5min"。
        max_workers: int
            最大并发工作进程数。
        max_collector_count: int
            最大重试次数。
        delay: float
            请求延迟。
        check_data_length: int
            检查数据长度。
        limit_nums: int
            用于调试，限制处理的股票数量。
        """
        # 登录baostock
        bs.login()
        super(BaostockCollectorHS3005min, self).__init__(
            save_dir=save_dir,
            start=start,
            end=end,
            interval=interval,
            max_workers=max_workers,
            max_collector_count=max_collector_count,
            delay=delay,
            check_data_length=check_data_length,
            limit_nums=limit_nums,
        )

    def get_trade_calendar(self):
        """获取指定时间范围内的交易日历。"""
        _format = "%Y-%m-%d"
        start = self.start_datetime.strftime(_format)
        end = self.end_datetime.strftime(_format)
        # 查询交易日数据
        rs = bs.query_trade_dates(start_date=start, end_date=end)
        calendar_list = []
        while (rs.error_code == "0") & rs.next():
            calendar_list.append(rs.get_row_data())
        calendar_df = pd.DataFrame(calendar_list, columns=rs.fields)
        # 筛选出交易日
        trade_calendar_df = calendar_df[~calendar_df["is_trading_day"].isin(["0"])]
        return trade_calendar_df["calendar_date"].values

    @staticmethod
    def process_interval(interval: str):
        """根据间隔参数，返回baostock API所需的参数。"""
        if interval == "1d":
            return {"interval": "d", "fields": "date,code,open,high,low,close,volume,amount,adjustflag"}
        if interval == "5min":
            return {"interval": "5", "fields": "date,time,code,open,high,low,close,volume,amount,adjustflag"}

    def get_data(
        self, symbol: str, interval: str, start_datetime: pd.Timestamp, end_datetime: pd.Timestamp
    ) -> pd.DataFrame:
        """
        获取单个股票的5分钟K线数据并进行初步处理。
        """
        df = self.get_data_from_remote(
            symbol=symbol, interval=interval, start_datetime=start_datetime, end_datetime=end_datetime
        )
        if df.empty:
            return pd.DataFrame()

        # 重命名列
        df.columns = ["date", "time", "symbol", "open", "high", "low", "close", "volume", "amount", "adjustflag"]
        # 合并time列并转换为datetime对象
        df["time"] = pd.to_datetime(df["time"], format="%Y%m%d%H%M%S%f")
        # 将time列作为新的date列，并调整为K线开始时间
        df["date"] = df["time"].dt.strftime("%Y-%m-%d %H:%M:%S")
        df["date"] = df["date"].map(lambda x: pd.Timestamp(x) - pd.Timedelta(minutes=5))
        df.drop(["time"], axis=1, inplace=True)
        # 规范化股票代码格式
        df["symbol"] = df["symbol"].map(lambda x: str(x).replace(".", "").upper())
        return df

    @staticmethod
    def get_data_from_remote(
        symbol: str, interval: str, start_datetime: pd.Timestamp, end_datetime: pd.Timestamp
    ) -> pd.DataFrame:
        """
        从baostock远程API获取K线数据。
        """
        df = pd.DataFrame()
        # 调用baostock接口
        rs = bs.query_history_k_data_plus(
            symbol,
            BaostockCollectorHS3005min.process_interval(interval=interval)["fields"],
            start_date=str(start_datetime.strftime("%Y-%m-%d")),
            end_date=str(end_datetime.strftime("%Y-%m-%d")),
            frequency=BaostockCollectorHS3005min.process_interval(interval=interval)["interval"],
            adjustflag="3",  # 使用不复权数据
        )
        if rs.error_code == "0" and len(rs.data) > 0:
            data_list = rs.data
            columns = rs.fields
            df = pd.DataFrame(data_list, columns=columns)
        return df

    def get_hs300_symbols(self) -> List[str]:
        """获取指定时间段内所有沪深300成分股的列表。"""
        hs300_stocks = []
        trade_calendar = self.get_trade_calendar()
        with tqdm(total=len(trade_calendar), desc="Fetching HS300 symbols") as p_bar:
            for date in trade_calendar:
                rs = bs.query_hs300_stocks(date=date)
                while rs.error_code == "0" and rs.next():
                    hs300_stocks.append(rs.get_row_data())
                p_bar.update()
        # 返回去重并排序后的股票代码列表
        return sorted({e[1] for e in hs300_stocks})

    def get_instrument_list(self):
        """获取要收集数据的股票列表（此处为沪深300成分股）。"""
        logger.info("开始获取沪深300成分股列表......")
        symbols = self.get_hs300_symbols()
        logger.info(f"获取到 {len(symbols)} 个股票代码。")
        return symbols

    def normalize_symbol(self, symbol: str):
        """规范化股票代码，例如 'sh.600000' -> 'SH600000'。"""
        return str(symbol).replace(".", "").upper()


class BaostockNormalizeHS3005min(BaseNormalize):
    """
    用于规范化从Baostock收集的5分钟K线数据。
    主要工作包括：对齐交易日历、计算复权价、计算涨跌幅等。
    """
    COLUMNS = ["open", "close", "high", "low", "volume"]
    # A股上午交易时间
    AM_RANGE = ("09:30:00", "11:29:00")
    # A股下午交易时间
    PM_RANGE = ("13:00:00", "14:59:00")

    def __init__(
        self, qlib_data_1d_dir: [str, Path], date_field_name: str = "date", symbol_field_name: str = "symbol", **kwargs
    ):
        """
        初始化。

        Parameters
        ----------
        qlib_data_1d_dir: str, Path
            Qlib格式的1日线数据目录。规范化5分钟数据需要用到日线数据的复权因子和停牌信息。
        """
        bs.login() # 登录baostock
        qlib.init(provider_uri=qlib_data_1d_dir) # 初始化qlib以读取日线数据
        # 加载所有股票的日线数据的 停牌、成交量、复权因子、收盘价 字段
        self.all_1d_data = D.features(D.instruments("all"), ["$paused", "$volume", "$factor", "$close"], freq="day")
        super(BaostockNormalizeHS3005min, self).__init__(date_field_name, symbol_field_name)

    @staticmethod
    def calc_change(df: pd.DataFrame, last_close: float) -> pd.Series:
        """计算涨跌幅。"""
        df = df.copy()
        _tmp_series = df["close"].ffill() # 向前填充收盘价以处理缺失值
        _tmp_shift_series = _tmp_series.shift(1) # 向下移动一位，得到上一周期的收盘价
        if last_close is not None:
            _tmp_shift_series.iloc[0] = float(last_close) # 使用上一个交易日的收盘价作为第一条记录的前收盘价
        change_series = _tmp_series / _tmp_shift_series - 1
        return change_series

    def _get_calendar_list(self) -> Iterable[pd.Timestamp]:
        """获取5分钟级别的交易日历。"""
        return self.generate_5min_from_daily(self.calendar_list_1d)

    @property
    def calendar_list_1d(self):
        """获取日线级别的交易日历（带缓存）。"""
        calendar_list_1d = getattr(self, "_calendar_list_1d", None)
        if calendar_list_1d is None:
            calendar_list_1d = self._get_1d_calendar_list()
            setattr(self, "_calendar_list_1d", calendar_list_1d)
        return calendar_list_1d

    @staticmethod
    def normalize_baostock(
        df: pd.DataFrame,
        calendar_list: list = None,
        date_field_name: str = "date",
        symbol_field_name: str = "symbol",
        last_close: float = None,
    ):
        """对单只股票的原始数据进行规范化。"""
        if df.empty:
            return df
        symbol = df.loc[df[symbol_field_name].first_valid_index(), symbol_field_name]
        df = df.copy()
        df.set_index(date_field_name, inplace=True)
        df.index = pd.to_datetime(df.index)
        df = df[~df.index.duplicated(keep="first")] # 去除重复的索引

        # 将数据与完整的5分钟日历对齐，缺失的分钟会被填充为NaN
        if calendar_list is not None:
            # 只对齐数据存在的时间范围内的日历
            start_time = pd.Timestamp(df.index.min()).date()
            end_time = pd.Timestamp(df.index.max()).date() + pd.Timedelta(days=1)
            sub_calendar = pd.DataFrame(index=calendar_list).loc[start_time:end_time].index
            df = df.reindex(sub_calendar)

        df.sort_index(inplace=True)
        # 成交量为0或NaN时，认为该分钟没有交易，将其他字段也设为NaN
        df.loc[(df["volume"] <= 0) | np.isnan(df["volume"]), list(set(df.columns) - {symbol_field_name})] = np.nan

        # 计算涨跌幅
        df["change"] = BaostockNormalizeHS3005min.calc_change(df, last_close)

        # 再次处理成交量为0的情况
        columns = copy.deepcopy(BaostockNormalizeHS3005min.COLUMNS) + ["change"]
        df.loc[(df["volume"] <= 0) | np.isnan(df["volume"]), columns] = np.nan

        df[symbol_field_name] = symbol
        df.index.names = [date_field_name]
        return df.reset_index()

    def generate_5min_from_daily(self, calendars: Iterable) -> pd.Index:
        """根据日线日历生成5分钟线日历。"""
        return generate_minutes_calendar_from_daily(
            calendals, freq="5min", am_range=self.AM_RANGE, pm_range=self.PM_RANGE
        )

    def adjusted_price(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        使用日线数据的复权因子计算复权价，并添加停牌信息。
        """
        df = calc_adjusted_price(
            df=df,
            _date_field_name=self._date_field_name,
            _symbol_field_name=self._symbol_field_name,
            frequence="5min",
            _1d_data_all=self.all_1d_data,
        )
        return df

    def _get_1d_calendar_list(self) -> Iterable[pd.Timestamp]:
        """从Qlib获取日线交易日历。"""
        return list(D.calendar(freq="day"))

    def normalize(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        执行完整的规范化流程。
        """
        # 步骤1: 基础规范化（对齐日历、计算涨跌幅等）
        df = self.normalize_baostock(df, self._calendar_list, self._date_field_name, self._symbol_field_name)
        # 步骤2: 计算复权价和停牌信息
        df = self.adjusted_price(df)
        return df


class Run(BaseRun):
    """
    用于将数据收集和规范化流程暴露为命令行接口的类。
    """
    def __init__(self, source_dir=None, normalize_dir=None, max_workers=1, interval="5min", region="HS300"):
        super().__init__(source_dir, normalize_dir, max_workers, interval)
        self.region = region

    @property
    def collector_class_name(self):
        """返回具体的收集器类名。"""
        return f"BaostockCollector{self.region.upper()}{self.interval}"

    @property
    def normalize_class_name(self):
        """返回具体的规范化类名。"""
        return f"BaostockNormalize{self.region.upper()}{self.interval}"

    @property
    def default_base_dir(self) -> [Path, str]:
        """返回默认的基础目录。"""
        return CUR_DIR

    def download_data(
        self,
        max_collector_count=2,
        delay=0.5,
        start=None,
        end=None,
        check_data_length=None,
        limit_nums=None,
    ):
        """
        从Baostock下载数据的命令行接口。
        """
        super(Run, self).download_data(max_collector_count, delay, start, end, check_data_length, limit_nums)

    def normalize_data(
        self,
        date_field_name: str = "date",
        symbol_field_name: str = "symbol",
        end_date: str = None,
        qlib_data_1d_dir: str = None,
    ):
        """
        规范化数据的命令行接口。

        注意
        ---------
        `qlib_data_1d_dir` 参数不能为空, 因为规范化5分钟数据需要使用到日线数据。
        """
        if qlib_data_1d_dir is None or not Path(qlib_data_1d_dir).expanduser().exists():
            raise ValueError(
                "规范化5分钟数据时，必须提供 `qlib_data_1d_dir` 参数，指向Qlib格式的日线数据目录。"
            )
        super(Run, self).normalize_data(
            date_field_name, symbol_field_name, end_date=end_date, qlib_data_1d_dir=qlib_data_1d_dir
        )


if __name__ == "__main__":
    fire.Fire(Run)
