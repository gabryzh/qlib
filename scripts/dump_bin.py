# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

# 导入abc模块，用于定义抽象基类
import abc
# 导入shutil模块，用于文件操作（如复制）
import shutil
# 导入traceback模块，用于追踪和打印异常信息
import traceback
# 从pathlib导入Path类，用于面向对象的文件系统路径操作
from pathlib import Path
# 从typing导入Iterable, List, Union，用于类型注解
from typing import Iterable, List, Union
# 从functools导入partial，用于创建偏函数
from functools import partial
# 导入concurrent.futures，用于并发执行任务
from concurrent.futures import ThreadPoolExecutor, as_completed, ProcessPoolExecutor

# 导入fire库，用于快速创建命令行界面
import fire
# 导入numpy库，用于数值计算
import numpy as np
# 导入pandas库，用于数据处理和分析
import pandas as pd
# 导入tqdm库，用于显示进度条
from tqdm import tqdm
# 导入loguru库，用于日志记录
from loguru import logger
# 从qlib.utils导入工具函数
from qlib.utils import fname_to_code, code_to_fname


def read_as_df(file_path: Union[str, Path], **kwargs) -> pd.DataFrame:
    """
    将csv或parquet文件读入pandas DataFrame。

    Parameters
    ----------
    file_path : Union[str, Path]
        数据文件的路径。
    **kwargs :
        传递给底层pandas读取器的其他关键字参数。

    Returns
    -------
    pd.DataFrame
        读取的数据帧。
    """
    # 扩展用户目录（例如'~'）并转换为Path对象
    file_path = Path(file_path).expanduser()
    # 获取文件后缀名并转换为小写
    suffix = file_path.suffix.lower()

    # 为不同文件类型保留特定的pandas读取参数
    keep_keys = {".csv": ("low_memory",)}
    kept_kwargs = {}
    for k in keep_keys.get(suffix, []):
        if k in kwargs:
            kept_kwargs[k] = kwargs[k]

    if suffix == ".csv":
        # 如果是csv文件，使用pd.read_csv读取
        return pd.read_csv(file_path, **kept_kwargs)
    elif suffix == ".parquet":
        # 如果是parquet文件，使用pd.read_parquet读取
        return pd.read_parquet(file_path, **kept_kwargs)
    else:
        # 如果是不支持的文件格式，抛出ValueError
        raise ValueError(f"不支持的文件格式: {suffix}")


class DumpDataBase:
    """数据转储的基类，定义了通用的常量和方法。"""
    # instruments文件中表示开始日期的字段名
    INSTRUMENTS_START_FIELD = "start_datetime"
    # instruments文件中表示结束日期的字段名
    INSTRUMENTS_END_FIELD = "end_datetime"
    # qlib数据目录中存放日历的文件夹名称
    CALENDARS_DIR_NAME = "calendars"
    # qlib数据目录中存放特征（因子）的文件夹名称
    FEATURES_DIR_NAME = "features"
    # qlib数据目录中存放股票列表的文件夹名称
    INSTRUMENTS_DIR_NAME = "instruments"
    # 转储的二进制文件的后缀名
    DUMP_FILE_SUFFIX = ".bin"
    # 日频数据的日期格式
    DAILY_FORMAT = "%Y-%m-%d"
    # 高频数据的日期时间格式
    HIGH_FREQ_FORMAT = "%Y-%m-%d %H:%M:%S"
    # instruments文件中的字段分隔符
    INSTRUMENTS_SEP = "\t"
    # instruments文件的名称
    INSTRUMENTS_FILE_NAME = "all.txt"

    # 更新模式：在现有数据基础上追加
    UPDATE_MODE = "update"
    # 全量模式：从头开始转储所有数据
    ALL_MODE = "all"

    def __init__(
        self,
        data_path: str,
        qlib_dir: str,
        backup_dir: str = None,
        freq: str = "day",
        max_workers: int = 16,
        date_field_name: str = "date",
        file_suffix: str = ".csv",
        symbol_field_name: str = "symbol",
        exclude_fields: str = "",
        include_fields: str = "",
        limit_nums: int = None,
    ):
        """
        初始化。

        Parameters
        ----------
        data_path: str
            股票数据的路径或目录。
        qlib_dir: str
            qlib数据（转储后）的目录。
        backup_dir: str, default None
            如果提供，则将qlib_dir备份到此目录。
        freq: str, default "day"
            交易频率（"day"或"1min"等）。
        max_workers: int, default 16
            用于并行处理的最大工作进程/线程数。
        date_field_name: str, default "date"
            CSV文件中日期字段的名称。
        file_suffix: str, default ".csv"
            数据文件的后缀名。
        symbol_field_name: str, default "symbol"
            CSV文件中股票代码字段的名称。
        exclude_fields: str
            需要排除不转储的字段，以逗号分隔。
        include_fields: str
            需要包含转储的字段，以逗号分隔。如果设置了此项，则只转储这些字段。
        limit_nums: int
            用于调试，限制处理的文件数量。默认为None，处理所有文件。
        """
        data_path = Path(data_path).expanduser()
        # 处理排除和包含字段
        if isinstance(exclude_fields, str):
            exclude_fields = exclude_fields.split(",")
        if isinstance(include_fields, str):
            include_fields = include_fields.split(",")
        self._exclude_fields = tuple(filter(lambda x: len(x) > 0, map(str.strip, exclude_fields)))
        self._include_fields = tuple(filter(lambda x: len(x) > 0, map(str.strip, include_fields)))

        self.file_suffix = file_suffix
        self.symbol_field_name = symbol_field_name
        # 获取所有源数据文件的路径
        self.df_files = sorted(data_path.glob(f"*{self.file_suffix}") if data_path.is_dir() else [data_path])
        if limit_nums is not None:
            self.df_files = self.df_files[: int(limit_nums)]

        self.qlib_dir = Path(qlib_dir).expanduser()
        self.backup_dir = backup_dir if backup_dir is None else Path(backup_dir).expanduser()
        if backup_dir is not None:
            self._backup_qlib_dir(Path(backup_dir).expanduser())

        self.freq = freq
        self.calendar_format = self.DAILY_FORMAT if self.freq == "day" else self.HIGH_FREQ_FORMAT

        self.works = max_workers
        self.date_field_name = date_field_name

        # 定义qlib数据目录下的各个子目录路径
        self._calendars_dir = self.qlib_dir.joinpath(self.CALENDARS_DIR_NAME)
        self._features_dir = self.qlib_dir.joinpath(self.FEATURES_DIR_NAME)
        self._instruments_dir = self.qlib_dir.joinpath(self.INSTRUMENTS_DIR_NAME)

        self._calendars_list = []

        self._mode = self.ALL_MODE
        self._kwargs = {}

    def _backup_qlib_dir(self, target_dir: Path):
        """备份qlib目录。"""
        shutil.copytree(str(self.qlib_dir.resolve()), str(target_dir.resolve()))

    def _format_datetime(self, datetime_d: [str, pd.Timestamp]):
        """格式化日期时间对象为字符串。"""
        datetime_d = pd.Timestamp(datetime_d)
        return datetime_d.strftime(self.calendar_format)

    def _get_date(
        self, file_or_df: [Path, pd.DataFrame], *, is_begin_end: bool = False, as_set: bool = False
    ) -> Iterable[pd.Timestamp]:
        """从文件或DataFrame中提取日期。"""
        if not isinstance(file_or_df, pd.DataFrame):
            df = self._get_source_data(file_or_df)
        else:
            df = file_or_df

        if df.empty or self.date_field_name not in df.columns.tolist():
            _calendars = pd.Series(dtype=np.float32)
        else:
            _calendars = df[self.date_field_name]

        # 根据参数返回不同形式的日期数据
        if is_begin_end and as_set:
            return (_calendars.min(), _calendars.max()), set(_calendars)
        elif is_begin_end:
            return _calendars.min(), _calendars.max()
        elif as_set:
            return set(_calendars)
        else:
            return _calendars.tolist()

    def _get_source_data(self, file_path: Path) -> pd.DataFrame:
        """读取源数据文件。"""
        df = read_as_df(file_path, low_memory=False)
        if self.date_field_name in df.columns:
            df[self.date_field_name] = pd.to_datetime(df[self.date_field_name])
        return df

    def get_symbol_from_file(self, file_path: Path) -> str:
        """从文件名中提取股票代码。"""
        return fname_to_code(file_path.stem.strip().lower())

    def get_dump_fields(self, df_columns: Iterable[str]) -> Iterable[str]:
        """根据include和exclude规则，确定需要转储的字段。"""
        if self._include_fields:
            return self._include_fields
        elif self._exclude_fields:
            return set(df_columns) - set(self._exclude_fields)
        else:
            return df_columns

    @staticmethod
    def _read_calendars(calendar_path: Path) -> List[pd.Timestamp]:
        """读取日历文件。"""
        return sorted(
            map(
                pd.Timestamp,
                pd.read_csv(calendar_path, header=None).loc[:, 0].tolist(),
            )
        )

    def _read_instruments(self, instrument_path: Path) -> pd.DataFrame:
        """读取instruments文件 (all.txt)。"""
        df = pd.read_csv(
            instrument_path,
            sep=self.INSTRUMENTS_SEP,
            names=[
                self.symbol_field_name,
                self.INSTRUMENTS_START_FIELD,
                self.INSTRUMENTS_END_FIELD,
            ],
        )
        return df

    def save_calendars(self, calendars_data: list):
        """保存日历文件。"""
        self._calendars_dir.mkdir(parents=True, exist_ok=True)
        calendars_path = str(self._calendars_dir.joinpath(f"{self.freq}.txt").expanduser().resolve())
        result_calendars_list = [self._format_datetime(x) for x in calendars_data]
        np.savetxt(calendars_path, result_calendars_list, fmt="%s", encoding="utf-8")

    def save_instruments(self, instruments_data: Union[list, pd.DataFrame]):
        """保存instruments文件 (all.txt)。"""
        self._instruments_dir.mkdir(parents=True, exist_ok=True)
        instruments_path = str(self._instruments_dir.joinpath(self.INSTRUMENTS_FILE_NAME).resolve())
        if isinstance(instruments_data, pd.DataFrame):
            _df_fields = [self.symbol_field_name, self.INSTRUMENTS_START_FIELD, self.INSTRUMENTS_END_FIELD]
            instruments_data = instruments_data.loc[:, _df_fields]
            instruments_data[self.symbol_field_name] = instruments_data[self.symbol_field_name].apply(
                lambda x: fname_to_code(x.lower()).upper()
            )
            instruments_data.to_csv(instruments_path, header=False, sep=self.INSTRUMENTS_SEP, index=False)
        else:
            np.savetxt(instruments_path, instruments_data, fmt="%s", encoding="utf-8")

    def data_merge_calendar(self, df: pd.DataFrame, calendars_list: List[pd.Timestamp]) -> pd.DataFrame:
        """将数据与日历对齐，填充缺失日期的NaN。"""
        calendars_df = pd.DataFrame(data=calendars_list, columns=[self.date_field_name])
        calendars_df[self.date_field_name] = calendars_df[self.date_field_name].astype("datetime64[ns]")
        cal_df = calendars_df[
            (calendars_df[self.date_field_name] >= df[self.date_field_name].min())
            & (calendars_df[self.date_field_name] <= df[self.date_field_name].max())
        ]
        cal_df.set_index(self.date_field_name, inplace=True)
        df.set_index(self.date_field_name, inplace=True)
        r_df = df.reindex(cal_df.index)
        return r_df

    @staticmethod
    def get_datetime_index(df: pd.DataFrame, calendar_list: List[pd.Timestamp]) -> int:
        """获取数据在全局日历中的起始位置索引。"""
        return calendar_list.index(df.index.min())

    def _data_to_bin(self, df: pd.DataFrame, calendar_list: List[pd.Timestamp], features_dir: Path):
        """将DataFrame数据转换为二进制格式并保存。"""
        if df.empty:
            logger.warning(f"{features_dir.name} 的数据为空或不存在")
            return
        if not calendar_list:
            logger.warning("日历列表为空")
            return

        # 与日历对齐
        _df = self.data_merge_calendar(df, calendar_list)
        if _df.empty:
            logger.warning(f"{features_dir.name} 的数据不在日历范围内")
            return

        # 获取起始日期索引
        date_index = self.get_datetime_index(_df, calendar_list)
        # 遍历每个字段并保存为.bin文件
        for field in self.get_dump_fields(_df.columns):
            bin_path = features_dir.joinpath(f"{field.lower()}.{self.freq}{self.DUMP_FILE_SUFFIX}")
            if field not in _df.columns:
                continue

            # 如果是更新模式且文件已存在，则以追加模式写入
            if bin_path.exists() and self._mode == self.UPDATE_MODE:
                with bin_path.open("ab") as fp:
                    np.array(_df[field]).astype("<f").tofile(fp)
            else:
                # 全量模式或文件不存在，则创建新文件并写入（包含起始索引）
                np.hstack([date_index, _df[field]]).astype("<f").tofile(str(bin_path.resolve()))

    def _dump_bin(self, file_or_data: [Path, pd.DataFrame], calendar_list: List[pd.Timestamp]):
        """处理单个股票数据并调用_data_to_bin进行转储。"""
        if not calendar_list:
            logger.warning("日历列表为空")
            return

        # 从文件路径或DataFrame中获取股票代码和数据
        if isinstance(file_or_data, pd.DataFrame):
            if file_or_data.empty:
                return
            code = fname_to_code(str(file_or_data.iloc[0][self.symbol_field_name]).lower())
            df = file_or_data
        elif isinstance(file_or_data, Path):
            code = self.get_symbol_from_file(file_or_data)
            df = self._get_source_data(file_or_data)
        else:
            raise ValueError(f"不支持的类型 {type(file_or_data)}")

        if df is None or df.empty:
            logger.warning(f"{code} 的数据为空或不存在")
            return

        # 删除重复的日期行，否则重置索引时会报错
        df = df.drop_duplicates(self.date_field_name)

        # 创建该股票的特征存储目录
        features_dir = self._features_dir.joinpath(code_to_fname(code).lower())
        features_dir.mkdir(parents=True, exist_ok=True)
        # 调用核心转换函数
        self._data_to_bin(df, calendar_list, features_dir)

    @abc.abstractmethod
    def dump(self):
        """抽象方法，子类必须实现具体的转储逻辑。"""
        raise NotImplementedError("dump方法未实现!")

    def __call__(self, *args, **kwargs):
        """使类的实例可调用，直接执行dump方法。"""
        self.dump()


class DumpDataAll(DumpDataBase):
    """全量转储数据的实现类。"""
    def _get_all_date(self):
        """获取所有源文件中的所有日期和每个文件的日期范围。"""
        logger.info("开始获取所有日期......")
        all_datetime = set()
        date_range_list = []
        _fun = partial(self._get_date, as_set=True, is_begin_end=True)
        # 使用进程池并行处理
        with tqdm(total=len(self.df_files)) as p_bar:
            with ProcessPoolExecutor(max_workers=self.works) as executor:
                for file_path, ((_begin_time, _end_time), _set_calendars) in zip(
                    self.df_files, executor.map(_fun, self.df_files)
                ):
                    # 合并所有唯一日期
                    all_datetime.update(_set_calendars)
                    if isinstance(_begin_time, pd.Timestamp) and isinstance(_end_time, pd.Timestamp):
                        # 格式化日期范围并添加到列表
                        _begin_time = self._format_datetime(_begin_time)
                        _end_time = self._format_datetime(_end_time)
                        symbol = self.get_symbol_from_file(file_path)
                        _inst_fields = [symbol.upper(), _begin_time, _end_time]
                        date_range_list.append(f"{self.INSTRUMENTS_SEP.join(_inst_fields)}")
                    p_bar.update()
        # 将结果存储在_kwargs中以供后续步骤使用
        self._kwargs["all_datetime_set"] = all_datetime
        self._kwargs["date_range_list"] = date_range_list
        logger.info("获取所有日期结束。\n")

    def _dump_calendars(self):
        """转储全局日历文件。"""
        logger.info("开始转储日历......")
        self._calendars_list = sorted(map(pd.Timestamp, self._kwargs["all_datetime_set"]))
        self.save_calendars(self._calendars_list)
        logger.info("日历转储结束。\n")

    def _dump_instruments(self):
        """转储instruments文件 (all.txt)。"""
        logger.info("开始转储instruments......")
        self.save_instruments(self._kwargs["date_range_list"])
        logger.info("instruments转储结束。\n")

    def _dump_features(self):
        """转储所有股票的特征数据。"""
        logger.info("开始转储特征......")
        _dump_func = partial(self._dump_bin, calendar_list=self._calendars_list)
        # 使用进程池并行处理
        with tqdm(total=len(self.df_files)) as p_bar:
            with ProcessPoolExecutor(max_workers=self.works) as executor:
                for _ in executor.map(_dump_func, self.df_files):
                    p_bar.update()

        logger.info("特征转储结束。\n")

    def dump(self):
        """执行全量转储的完整流程。"""
        self._get_all_date()
        self._dump_calendars()
        self._dump_instruments()
        self._dump_features()


class DumpDataFix(DumpDataAll):
    """
    修复数据的实现类，用于向现有qlib数据中添加新股票，但不重建日历。
    """
    def _dump_instruments(self):
        """更新instruments文件，添加新股票的日期范围。"""
        logger.info("开始转储instruments......")
        _fun = partial(self._get_date, is_begin_end=True)
        # 找出不在现有instruments文件中的新股票
        new_stock_files = sorted(
            filter(
                lambda x: self.get_symbol_from_file(x).upper() not in self._old_instruments,
                self.df_files,
            )
        )
        with tqdm(total=len(new_stock_files)) as p_bar:
            with ProcessPoolExecutor(max_workers=self.works) as execute:
                for file_path, (_begin_time, _end_time) in zip(new_stock_files, execute.map(_fun, new_stock_files)):
                    if isinstance(_begin_time, pd.Timestamp) and isinstance(_end_time, pd.Timestamp):
                        symbol = self.get_symbol_from_file(file_path).upper()
                        _dt_map = self._old_instruments.setdefault(symbol, dict())
                        _dt_map[self.INSTRUMENTS_START_FIELD] = self._format_datetime(_begin_time)
                        _dt_map[self.INSTRUMENTS_END_FIELD] = self._format_datetime(_end_time)
                    p_bar.update()
        # 将更新后的instruments数据保存回文件
        _inst_df = pd.DataFrame.from_dict(self._old_instruments, orient="index")
        _inst_df.index.names = [self.symbol_field_name]
        self.save_instruments(_inst_df.reset_index())
        logger.info("instruments转储结束。\n")

    def dump(self):
        """执行修复流程。"""
        # 读取现有日历
        self._calendars_list = self._read_calendars(self._calendars_dir.joinpath(f"{self.freq}.txt"))
        # 读取现有instruments
        self._old_instruments = (
            self._read_instruments(self._instruments_dir.joinpath(self.INSTRUMENTS_FILE_NAME))
            .set_index([self.symbol_field_name])
            .to_dict(orient="index")
        )
        # 更新instruments和转储新特征
        self._dump_instruments()
        self._dump_features()


class DumpDataUpdate(DumpDataBase):
    """增量更新数据的实现类。"""
    def __init__(
        self,
        # ... (参数与基类相同)
        data_path: str,
        qlib_dir: str,
        backup_dir: str = None,
        freq: str = "day",
        max_workers: int = 16,
        date_field_name: str = "date",
        file_suffix: str = ".csv",
        symbol_field_name: str = "symbol",
        exclude_fields: str = "",
        include_fields: str = "",
        limit_nums: int = None,
    ):
        super().__init__(
            data_path, qlib_dir, backup_dir, freq, max_workers, date_field_name,
            file_suffix, symbol_field_name, exclude_fields, include_fields, limit_nums,
        )
        self._mode = self.UPDATE_MODE
        # 读取旧日历和instruments
        self._old_calendar_list = self._read_calendars(self._calendars_dir.joinpath(f"{self.freq}.txt"))
        self._update_instruments = (
            self._read_instruments(self._instruments_dir.joinpath(self.INSTRUMENTS_FILE_NAME))
            .set_index([self.symbol_field_name])
            .to_dict(orient="index")
        )

        # 加载所有源数据到内存（注意：可能需要大量内存）
        self._all_data = self._load_all_source_data()
        # 生成新的日历（旧日历 + 新数据中的新日期）
        self._new_calendar_list = self._old_calendar_list + sorted(
            list(filter(lambda x: x > self._old_calendar_list[-1], pd.to_datetime(self._all_data[self.date_field_name]).unique()))
        )


    def _load_all_source_data(self) -> pd.DataFrame:
        """加载所有源数据文件到一个DataFrame中。"""
        logger.info("开始加载所有源数据....")
        all_df = []

        def _read_df(file_path: Path):
            _df = read_as_df(file_path)
            if self.date_field_name in _df.columns and not np.issubdtype(
                _df[self.date_field_name].dtype, np.datetime64
            ):
                _df[self.date_field_name] = pd.to_datetime(_df[self.date_field_name])
            if self.symbol_field_name not in _df.columns:
                _df[self.symbol_field_name] = self.get_symbol_from_file(file_path)
            return _df

        # 使用线程池并行读取
        with tqdm(total=len(self.df_files)) as p_bar:
            with ThreadPoolExecutor(max_workers=self.works) as executor:
                for df in executor.map(_read_df, self.df_files):
                    if not df.empty:
                        all_df.append(df)
                    p_bar.update()

        logger.info("加载所有数据结束。\n")
        return pd.concat(all_df, sort=False)

    def _dump_calendars(self):
        """在更新模式下，日历的保存被移到主dump流程中。"""
        pass

    def _dump_instruments(self):
        """在更新模式下，instruments的保存被移到主dump流程中。"""
        pass

    def _dump_features(self):
        """转储特征数据，区分新股票和现有股票的追加数据。"""
        logger.info("开始转储特征......")
        error_code = {}
        with ProcessPoolExecutor(max_workers=self.works) as executor:
            futures = {}
            # 按股票代码分组处理
            for _code, _df in self._all_data.groupby(self.symbol_field_name, group_keys=False):
                _code = fname_to_code(str(_code).lower()).upper()
                _start, _end = self._get_date(_df, is_begin_end=True)
                if not (isinstance(_start, pd.Timestamp) and isinstance(_end, pd.Timestamp)):
                    continue

                if _code in self._update_instruments:
                    # 对于已存在的股票，找出新的日期数据并追加
                    # 注意：这里假设日期是连续的，并且新数据都在旧数据之后
                    _end_date_in_instruments = self._update_instruments[_code][self.INSTRUMENTS_END_FIELD]
                    _update_df = _df[_df[self.date_field_name] > pd.Timestamp(_end_date_in_instruments)]

                    if not _update_df.empty:
                        self._update_instruments[_code][self.INSTRUMENTS_END_FIELD] = self._format_datetime(_end)
                        futures[executor.submit(self._dump_bin, _update_df, self._new_calendar_list)] = _code
                else:
                    # 对于新股票，全量转储其数据
                    _dt_range = self._update_instruments.setdefault(_code, dict())
                    _dt_range[self.INSTRUMENTS_START_FIELD] = self._format_datetime(_start)
                    _dt_range[self.INSTRUMENTS_END_FIELD] = self._format_datetime(_end)
                    futures[executor.submit(self._dump_bin, _df, self. _new_calendar_list)] = _code

            # 等待所有任务完成并处理异常
            with tqdm(total=len(futures)) as p_bar:
                for _future in as_completed(futures):
                    try:
                        _future.result()
                    except Exception:
                        error_code[futures[_future]] = traceback.format_exc()
                    p_bar.update()
            logger.info(f"转储bin文件时的错误: {error_code}")

        logger.info("特征转储结束。\n")

    def dump(self):
        """执行增量更新的完整流程。"""
        # 1. 保存包含新日期的完整日历
        self.save_calendars(self._new_calendar_list)
        # 2. 转储特征数据（追加或新建）
        self._dump_features()
        # 3. 保存更新后的instruments信息
        df = pd.DataFrame.from_dict(self._update_instruments, orient="index")
        df.index.names = [self.symbol_field_name]
        self.save_instruments(df.reset_index())


if __name__ == "__main__":
    # 使用fire库将不同的转储类暴露为命令行命令
    fire.Fire({"dump_all": DumpDataAll, "dump_fix": DumpDataFix, "dump_update": DumpDataUpdate})
