# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

# 导入abc模块，用于定义抽象基类
import abc
# 导入time模块，用于处理时间，例如延时
import time
# 导入datetime模块，用于处理日期和时间
import datetime
# 导入importlib模块，用于动态导入模块
import importlib
# 从pathlib导入Path类，用于面向对象的文件系统路径
from pathlib import Path
# 从typing导入Type和Iterable，用于类型注解
from typing import Type, Iterable
# 从concurrent.futures导入ProcessPoolExecutor，用于进程池并发
from concurrent.futures import ProcessPoolExecutor

# 导入pandas库，用于数据处理和分析
import pandas as pd
# 导入tqdm库，用于显示进度条
from tqdm import tqdm
# 导入loguru库，用于日志记录
from loguru import logger
# 从joblib导入Parallel和delayed，用于简单的并行计算
from joblib import Parallel, delayed
# 从qlib.utils导入code_to_fname，用于将股票代码转换为文件名
from qlib.utils import code_to_fname


class BaseCollector(abc.ABC):
    """
    数据收集器的抽象基类。
    所有具体的数据收集器（如从Yahoo, Baidu等）都应继承此类，并实现其抽象方法。
    """
    # 缓存标志：表示数据因过短而被缓存
    CACHE_FLAG = "CACHED"
    # 正常标志：表示数据正常处理
    NORMAL_FLAG = "NORMAL"

    # 日线数据的默认开始时间
    DEFAULT_START_DATETIME_1D = pd.Timestamp("2000-01-01")
    # 分钟线数据的默认开始时间（默认为大约5个交易周前）
    DEFAULT_START_DATETIME_1MIN = pd.Timestamp(datetime.datetime.now() - pd.Timedelta(days=5 * 6 - 1)).date()
    # 日线数据的默认结束时间（默认为明天）
    DEFAULT_END_DATETIME_1D = pd.Timestamp(datetime.datetime.now() + pd.Timedelta(days=1)).date()
    # 分钟线数据的默认结束时间
    DEFAULT_END_DATETIME_1MIN = DEFAULT_END_DATETIME_1D

    # 数据频率常量
    INTERVAL_1min = "1min"
    INTERVAL_1d = "1d"

    def __init__(
        self,
        save_dir: [str, Path],
        start=None,
        end=None,
        interval="1d",
        max_workers=1,
        max_collector_count=2,
        delay=0,
        check_data_length: int = None,
        limit_nums: int = None,
    ):
        """
        初始化。

        Parameters
        ----------
        save_dir: str or Path
            数据保存目录。
        start: str or pd.Timestamp
            开始时间。
        end: str or pd.Timestamp
            结束时间。
        interval: str
            数据频率, 可选值为 ["1min", "1d"]。
        max_workers: int
            并发工作进程数。在收集数据时，建议设为1以避免IP被封。
        max_collector_count: int
            对于获取数据失败或数据长度不足的股票，最大重试次数。
        delay: float
            每次请求之间的延迟（秒）。
        check_data_length: int
            检查数据长度。如果设置，数据长度小于此值的股票将被视为不完整，并会进行重试。
        limit_nums: int
            用于调试，限制处理的股票数量。
        """
        self.save_dir = Path(save_dir).expanduser().resolve()
        self.save_dir.mkdir(parents=True, exist_ok=True)

        self.delay = delay
        self.max_workers = max_workers
        self.max_collector_count = max_collector_count
        # 用于缓存数据长度过短的股票数据
        self.mini_symbol_map = {}
        self.interval = interval
        self.check_data_length = max(int(check_data_length) if check_data_length is not None else 0, 0)

        # 规范化开始和结束时间
        self.start_datetime = self.normalize_start_datetime(start)
        self.end_datetime = self.normalize_end_datetime(end)

        # 获取并排序去重后的股票列表
        self.instrument_list = sorted(set(self.get_instrument_list()))

        if limit_nums is not None:
            try:
                self.instrument_list = self.instrument_list[: int(limit_nums)]
            except Exception as e:
                logger.warning(f"无法使用 limit_nums={limit_nums}, 该参数将被忽略")

    def normalize_start_datetime(self, start_datetime: [str, pd.Timestamp] = None):
        """规范化开始时间，如果未提供则使用默认值。"""
        return (
            pd.Timestamp(str(start_datetime))
            if start_datetime
            else getattr(self, f"DEFAULT_START_DATETIME_{self.interval.upper()}")
        )

    def normalize_end_datetime(self, end_datetime: [str, pd.Timestamp] = None):
        """规范化结束时间，如果未提供则使用默认值。"""
        return (
            pd.Timestamp(str(end_datetime))
            if end_datetime
            else getattr(self, f"DEFAULT_END_DATETIME_{self.interval.upper()}")
        )

    @abc.abstractmethod
    def get_instrument_list(self):
        """抽象方法：获取股票列表。子类必须实现。"""
        raise NotImplementedError("必须重写 get_instrument_list 方法")

    @abc.abstractmethod
    def normalize_symbol(self, symbol: str):
        """抽象方法：规范化股票代码。子类必须实现。"""
        raise NotImplementedError("必须重写 normalize_symbol 方法")

    @abc.abstractmethod
    def get_data(
        self, symbol: str, interval: str, start_datetime: pd.Timestamp, end_datetime: pd.Timestamp
    ) -> pd.DataFrame:
        """
        抽象方法：获取单个股票的数据。子类必须实现。

        Returns
        ---------
            pd.DataFrame: 必须包含 "symbol" 和 "date" 列。
        """
        raise NotImplementedError("必须重写 get_data 方法")

    def sleep(self):
        """根据设置的延迟时间进行休眠。"""
        time.sleep(self.delay)

    def _simple_collector(self, symbol: str):
        """单个股票的收集逻辑。"""
        self.sleep()
        df = self.get_data(symbol, self.interval, self.start_datetime, self.end_datetime)
        _result = self.NORMAL_FLAG
        if self.check_data_length > 0:
            # 如果开启了数据长度检查，则检查并可能缓存数据
            _result = self.cache_small_data(symbol, df)
        if _result == self.NORMAL_FLAG:
            # 如果数据正常，则保存
            self.save_instrument(symbol, df)
        return _result

    def save_instrument(self, symbol, df: pd.DataFrame):
        """
        将单个股票的数据保存到文件。

        Parameters
        ----------
        symbol: str
            股票代码。
        df : pd.DataFrame
            包含股票数据的DataFrame，必须有 "symbol" 和 "datetime" 列。
        """
        if df is None or df.empty:
            logger.warning(f"{symbol} 的数据为空")
            return

        symbol = self.normalize_symbol(symbol)
        # 将股票代码转换为安全的文件名格式
        symbol_fname = code_to_fname(symbol)
        instrument_path = self.save_dir.joinpath(f"{symbol_fname}.csv")
        df["symbol"] = symbol
        # 如果文件已存在，则追加数据
        if instrument_path.exists():
            _old_df = pd.read_csv(instrument_path)
            df = pd.concat([_old_df, df], sort=False)
        df.to_csv(instrument_path, index=False)

    def cache_small_data(self, symbol, df):
        """如果数据长度不足，则缓存数据以备后续合并。"""
        if len(df) < self.check_data_length:
            logger.warning(f"{symbol} 的交易日数少于 {self.check_data_length}!")
            _temp = self.mini_symbol_map.setdefault(symbol, [])
            _temp.append(df.copy())
            return self.CACHE_FLAG
        else:
            # 如果数据长度足够，且之前被缓存过，则从缓存中移除
            if symbol in self.mini_symbol_map:
                self.mini_symbol_map.pop(symbol)
            return self.NORMAL_FLAG

    def _collector(self, instrument_list):
        """对一个列表的股票进行数据收集。"""
        error_symbol = []
        # 使用joblib进行并行处理
        res = Parallel(n_jobs=self.max_workers)(
            delayed(self._simple_collector)(_inst) for _inst in tqdm(instrument_list)
        )
        for _symbol, _result in zip(instrument_list, res):
            if _result != self.NORMAL_FLAG:
                error_symbol.append(_symbol)

        logger.info(f"收集失败或数据过短的股票数量: {len(error_symbol)}")
        logger.info(f"当前轮次尝试收集的股票数量: {len(instrument_list)}")
        # 将当前轮次的失败列表与仍然在缓存中的股票合并，作为下一轮的输入
        error_symbol.extend(self.mini_symbol_map.keys())
        return sorted(set(error_symbol))

    def collector_data(self):
        """收集数据的总入口和调度逻辑。"""
        logger.info("开始收集数据......")
        instrument_list = self.instrument_list
        # 进行多轮收集，直到所有数据都成功获取或达到最大重试次数
        for i in range(self.max_collector_count):
            if not instrument_list:
                break
            logger.info(f"第 {i+1} 轮数据获取中...")
            instrument_list = self._collector(instrument_list)
            logger.info(f"第 {i+1} 轮获取结束。")

        # 处理最终仍在缓存中的（数据长度不足的）股票
        for _symbol, _df_list in self.mini_symbol_map.items():
            _df = pd.concat(_df_list, sort=False)
            if not _df.empty:
                self.save_instrument(_symbol, _df.drop_duplicates(["date"]).sort_values(["date"]))
        if self.mini_symbol_map:
            logger.warning(f"最终数据长度仍小于 {self.check_data_length} 的股票列表: {list(self.mini_symbol_map.keys())}")

        logger.info(f"总共 {len(self.instrument_list)} 个股票, 最终失败: {len(set(instrument_list))} 个")


class BaseNormalize(abc.ABC):
    """数据规范化的抽象基类。"""
    def __init__(self, date_field_name: str = "date", symbol_field_name: str = "symbol", **kwargs):
        self._date_field_name = date_field_name
        self._symbol_field_name = symbol_field_name
        self.kwargs = kwargs
        # 获取基准日历
        self._calendar_list = self._get_calendar_list()

    @abc.abstractmethod
    def normalize(self, df: pd.DataFrame) -> pd.DataFrame:
        """抽象方法：规范化DataFrame。子类必须实现。"""
        raise NotImplementedError("必须重写 normalize 方法")

    @abc.abstractmethod
    def _get_calendar_list(self) -> Iterable[pd.Timestamp]:
        """抽象方法：获取基准日历列表。子类必须实现。"""
        raise NotImplementedError("必须重写 _get_calendar_list 方法")


class Normalize:
    """数据规范化的执行器类。"""
    def __init__(
        self,
        source_dir: [str, Path],
        target_dir: [str, Path],
        normalize_class: Type[BaseNormalize],
        max_workers: int = 16,
        date_field_name: str = "date",
        symbol_field_name: str = "symbol",
        **kwargs,
    ):
        """
        初始化。

        Parameters
        ----------
        source_dir: str or Path
            原始数据目录。
        target_dir: str or Path
            规范化后数据的存放目录。
        normalize_class: Type[BaseNormalize]
            具体的规范化逻辑实现类。
        max_workers: int
            最大并发进程数。
        date_field_name: str
            日期字段名。
        symbol_field_name: str
            股票代码字段名。
        """
        if not (source_dir and target_dir):
            raise ValueError("source_dir 和 target_dir 不能为空")
        self._source_dir = Path(source_dir).expanduser()
        self._target_dir = Path(target_dir).expanduser()
        self._target_dir.mkdir(parents=True, exist_ok=True)
        self._date_field_name = date_field_name
        self._symbol_field_name = symbol_field_name
        self._end_date = kwargs.get("end_date", None)
        self._max_workers = max_workers

        # 实例化具体的规范化类
        self._normalize_obj = normalize_class(
            date_field_name=date_field_name, symbol_field_name=symbol_field_name, **kwargs
        )

    def _executor(self, file_path: Path):
        """处理单个文件的规范化。"""
        file_path = Path(file_path)

        # 特殊处理：防止pandas将'NA'这样的股票代码误解析为NaN
        default_na = pd._libs.parsers.STR_NA_VALUES.copy()
        symbol_na = default_na.copy()
        if "NA" in symbol_na: symbol_na.remove("NA") # 'NA' might be a valid symbol

        columns = pd.read_csv(file_path, nrows=0).columns
        df = pd.read_csv(
            file_path,
            dtype={self._symbol_field_name: str}, # 强制股票代码列为字符串
            keep_default_na=False,
            na_values={col: symbol_na if col == self._symbol_field_name else default_na for col in columns},
        )

        # 调用具体的规范化逻辑
        df = self._normalize_obj.normalize(df)
        if df is not None and not df.empty:
            # 如果设置了结束日期，则裁剪数据
            if self._end_date is not None:
                _mask = pd.to_datetime(df[self._date_field_name]) <= pd.Timestamp(self._end_date)
                df = df[_mask]
            # 保存规范化后的数据
            df.to_csv(self._target_dir.joinpath(file_path.name), index=False)

    def normalize(self):
        """执行所有文件的规范化。"""
        logger.info("开始规范化数据......")

        with ProcessPoolExecutor(max_workers=self._max_workers) as worker:
            file_list = list(self._source_dir.glob("*.csv"))
            with tqdm(total=len(file_list)) as p_bar:
                # 并发处理所有文件
                for _ in worker.map(self._executor, file_list):
                    p_bar.update()


class BaseRun(abc.ABC):
    """
    将下载和规范化流程封装成命令行工具的抽象基类。
    """
    def __init__(self, source_dir=None, normalize_dir=None, max_workers=1, interval="1d"):
        if source_dir is None:
            source_dir = Path(self.default_base_dir).joinpath("source")
        self.source_dir = Path(source_dir).expanduser().resolve()
        self.source_dir.mkdir(parents=True, exist_ok=True)

        if normalize_dir is None:
            normalize_dir = Path(self.default_base_dir).joinpath("normalize")
        self.normalize_dir = Path(normalize_dir).expanduser().resolve()
        self.normalize_dir.mkdir(parents=True, exist_ok=True)

        # 动态导入包含具体实现的模块（通常是collector.py）
        self._cur_module = importlib.import_module("collector")
        self.max_workers = max_workers
        self.interval = interval

    @property
    @abc.abstractmethod
    def collector_class_name(self) -> str:
        """抽象属性：返回具体的收集器类名。子类必须实现。"""
        raise NotImplementedError("必须重写 collector_class_name")

    @property
    @abc.abstractmethod
    def normalize_class_name(self) -> str:
        """抽象属性：返回具体的规范化类名。子类必须实现。"""
        raise NotImplementedError("必须重写 normalize_class_name")

    @property
    @abc.abstractmethod
    def default_base_dir(self) -> [Path, str]:
        """抽象属性：返回默认的基础目录。子类必须实现。"""
        raise NotImplementedError("必须重写 default_base_dir")

    def download_data(
        self,
        max_collector_count=2,
        delay=0,
        start=None,
        end=None,
        check_data_length: int = None,
        limit_nums=None,
        **kwargs,
    ):
        """
        从互联网下载数据的命令行接口。
        """
        _class = getattr(self._cur_module, self.collector_class_name)
        _class(
            self.source_dir,
            max_workers=self.max_workers,
            max_collector_count=max_collector_count,
            delay=delay,
            start=start,
            end=end,
            interval=self.interval,
            check_data_length=check_data_length,
            limit_nums=limit_nums,
            **kwargs,
        ).collector_data()

    def normalize_data(self, date_field_name: str = "date", symbol_field_name: str = "symbol", **kwargs):
        """
        规范化数据的命令行接口。
        """
        _class = getattr(self._cur_module, self.normalize_class_name)
        yc = Normalize(
            source_dir=self.source_dir,
            target_dir=self.normalize_dir,
            normalize_class=_class,
            max_workers=self.max_workers,
            date_field_name=date_field_name,
            symbol_field_name=symbol_field_name,
            **kwargs,
        )
        yc.normalize()
