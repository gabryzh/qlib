# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.


from __future__ import division
from __future__ import print_function

import re
import abc
import copy
import queue
import bisect
import numpy as np
import pandas as pd
from typing import List, Union, Optional

# 为了支持外部代码中的多进程，这里使用了 joblib
from joblib import delayed

from .cache import H
from ..config import C
from .inst_processor import InstProcessor

from ..log import get_module_logger
from .cache import DiskDatasetCache
from ..utils import (
    Wrapper,
    init_instance_by_config,
    register_wrapper,
    get_module_by_module_path,
    parse_field,
    hash_args,
    normalize_cache_fields,
    code_to_fname,
    time_to_slc_point,
    read_period_data,
    get_period_list,
)
from ..utils.paral import ParallelExt
from .ops import Operators  # pylint: disable=W0611  # noqa: F401


class ProviderBackendMixin:
    """
    此辅助类旨在使基于存储后端的提供程序更加方便。
    如果提供程序不依赖于后端存储，则无需继承此类。
    """

    def get_default_backend(self):
        """获取默认的后端配置"""
        backend = {}
        provider_name: str = re.findall("[A-Z][^A-Z]*", self.__class__.__name__)[-2]
        # 设置默认存储类
        backend.setdefault("class", f"File{provider_name}Storage")
        # 设置默认存储模块路径
        backend.setdefault("module_path", "qlib.data.storage.file_storage")
        return backend

    def backend_obj(self, **kwargs):
        """根据配置初始化后端对象"""
        backend = self.backend if self.backend else self.get_default_backend()
        backend = copy.deepcopy(backend)
        backend.setdefault("kwargs", {}).update(**kwargs)
        return init_instance_by_config(backend)


class CalendarProvider(abc.ABC):
    """日历提供程序基类

    提供日历数据。
    """

    def calendar(self, start_time=None, end_time=None, freq="day", future=False):
        """在给定时间范围内获取特定市场的日历。

        参数
        ----------
        start_time : str
            时间范围的开始。
        end_time : str
            时间范围的结束。
        freq : str
            时间频率，可用值：year/quarter/month/week/day。
        future : bool
            是否包括未来的交易日。

        返回
        ----------
        list
            日历列表
        """
        _calendar, _calendar_index = self._get_calendar(freq, future)
        if start_time == "None":
            start_time = None
        if end_time == "None":
            end_time = None
        # 裁剪
        if start_time:
            start_time = pd.Timestamp(start_time)
            if start_time > _calendar[-1]:
                return np.array([])
        else:
            start_time = _calendar[0]
        if end_time:
            end_time = pd.Timestamp(end_time)
            if end_time < _calendar[0]:
                return np.array([])
        else:
            end_time = _calendar[-1]
        _, _, si, ei = self.locate_index(start_time, end_time, freq, future)
        return _calendar[si : ei + 1]

    def locate_index(
        self, start_time: Union[pd.Timestamp, str], end_time: Union[pd.Timestamp, str], freq: str, future: bool = False
    ):
        """在特定频率的日历中定位开始时间和结束时间的索引。

        参数
        ----------
        start_time : pd.Timestamp
            时间范围的开始。
        end_time : pd.Timestamp
            时间范围的结束。
        freq : str
            时间频率，可用值：year/quarter/month/week/day。
        future : bool
            是否包括未来的交易日。

        返回
        -------
        pd.Timestamp
            实际的开始时间。
        pd.Timestamp
            实际的结束时间。
        int
            开始时间的索引。
        int
            结束时间的索引。
        """
        start_time = pd.Timestamp(start_time)
        end_time = pd.Timestamp(end_time)
        calendar, calendar_index = self._get_calendar(freq=freq, future=future)
        if start_time not in calendar_index:
            try:
                start_time = calendar[bisect.bisect_left(calendar, start_time)]
            except IndexError as index_e:
                raise IndexError(
                    "`start_time` 使用了一个未来的日期，如果你想获取未来的交易日，可以使用 `future=True`"
                ) from index_e
        start_index = calendar_index[start_time]
        if end_time not in calendar_index:
            end_time = calendar[bisect.bisect_right(calendar, end_time) - 1]
        end_index = calendar_index[end_time]
        return start_time, end_time, start_index, end_index

    def _get_calendar(self, freq, future):
        """使用内存缓存加载日历。

        参数
        ----------
        freq : str
            读取日历文件的频率。
        future : bool
            是否包括未来的交易日。

        返回
        -------
        list
            时间戳列表。
        dict
            以时间戳为键、索引为值的字典，用于快速搜索。
        """
        flag = f"{freq}_future_{future}"
        if flag not in H["c"]:
            _calendar = np.array(self.load_calendar(freq, future))
            _calendar_index = {x: i for i, x in enumerate(_calendar)}  # 用于快速搜索
            H["c"][flag] = _calendar, _calendar_index
        return H["c"][flag]

    def _uri(self, start_time, end_time, freq, future=False):
        """获取日历生成任务的 URI。"""
        return hash_args(start_time, end_time, freq, future)

    def load_calendar(self, freq, future):
        """从文件加载原始日历时间戳。

        参数
        ----------
        freq : str
            读取日历文件的频率。
        future: bool

        返回
        ----------
        list
            时间戳列表
        """
        raise NotImplementedError("CalendarProvider 的子类必须实现 `load_calendar` 方法")


class InstrumentProvider(abc.ABC):
    """金融工具提供程序基类

    提供金融工具数据。
    """

    @staticmethod
    def instruments(market: Union[List, str] = "all", filter_pipe: Union[List, None] = None):
        """获取基础市场添加多个动态过滤器后的一般配置字典。

        参数
        ----------
        market : Union[List, str]
            str:
                市场/行业/指数简称，例如 all/sse/szse/sse50/csi300/csi500。
            list:
                ["ID1", "ID2"]。一个股票列表。
        filter_pipe : list
            动态过滤器列表。

        返回
        ----------
        dict: if isinstance(market, str)
            股票池配置字典。

            {`market` => 基础市场名称, `filter_pipe` => 过滤器列表}

            示例 :

            .. code-block::

                {'market': 'csi500',
                'filter_pipe': [{'filter_type': 'ExpressionDFilter',
                'rule_expression': '$open<40',
                'filter_start_time': None,
                'filter_end_time': None,
                'keep': False},
                {'filter_type': 'NameDFilter',
                'name_rule_re': 'SH[0-9]{4}55',
                'filter_start_time': None,
                'filter_end_time': None}]}

        list: if isinstance(market, list)
            直接返回原始列表。
            注意：这将使金融工具与更多情况兼容，用户代码将更简单。
        """
        if isinstance(market, list):
            return market
        from .filter import SeriesDFilter  # pylint: disable=C0415

        if filter_pipe is None:
            filter_pipe = []
        config = {"market": market, "filter_pipe": []}
        # 过滤器的顺序会影响结果，所以我们需要保持顺序
        for filter_t in filter_pipe:
            if isinstance(filter_t, dict):
                _config = filter_t
            elif isinstance(filter_t, SeriesDFilter):
                _config = filter_t.to_config()
            else:
                raise TypeError(
                    f"不支持的过滤器类型: {type(filter_t)}！过滤器只支持 dict 或 isinstance(filter, SeriesDFilter)"
                )
            config["filter_pipe"].append(_config)
        return config

    @abc.abstractmethod
    def list_instruments(self, instruments, start_time=None, end_time=None, freq="day", as_list=False):
        """根据特定的股票池配置列出金融工具。

        参数
        ----------
        instruments : dict
            股票池配置。
        start_time : str
            时间范围的开始。
        end_time : str
            时间范围的结束。
        as_list : bool
            以列表或字典形式返回金融工具。

        返回
        -------
        dict or list
            金融工具列表或带时间跨度的字典
        """
        raise NotImplementedError("InstrumentProvider 的子类必须实现 `list_instruments` 方法")

    def _uri(self, instruments, start_time=None, end_time=None, freq="day", as_list=False):
        return hash_args(instruments, start_time, end_time, freq, as_list)

    # instruments 类型
    LIST = "LIST"
    DICT = "DICT"
    CONF = "CONF"

    @classmethod
    def get_inst_type(cls, inst):
        if "market" in inst:
            return cls.CONF
        if isinstance(inst, dict):
            return cls.DICT
        if isinstance(inst, (list, tuple, pd.Index, np.ndarray)):
            return cls.LIST
        raise ValueError(f"未知的金融工具类型 {inst}")


class FeatureProvider(abc.ABC):
    """特征提供程序类

    提供特征数据。
    """

    @abc.abstractmethod
    def feature(self, instrument, field, start_time, end_time, freq):
        """获取特征数据。

        参数
        ----------
        instrument : str
            一个特定的金融工具。
        field : str
            特征的特定字段。
        start_time : str
            时间范围的开始。
        end_time : str
            时间范围的结束。
        freq : str
            时间频率，可用值：year/quarter/month/week/day。

        返回
        -------
        pd.Series
            特定特征的数据
        """
        raise NotImplementedError("FeatureProvider 的子类必须实现 `feature` 方法")


class PITProvider(abc.ABC):
    """即时（Point-In-Time）数据提供程序基类"""
    @abc.abstractmethod
    def period_feature(
        self,
        instrument,
        field,
        start_index: int,
        end_index: int,
        cur_time: pd.Timestamp,
        period: Optional[int] = None,
    ) -> pd.Series:
        """
        获取 `start_index` 和 `end_index` 之间的历史时期数据序列

        参数
        ----------
        start_index: int
            start_index 是相对于 cur_time 最新时期的相对索引

        end_index: int
            end_index 是相对于 cur_time 最新时期的相对索引
            在大多数情况下，start_index 和 end_index 都是非正值
            例如，start_index == -3, end_index == 0，当前时期索引为 cur_idx,
            那么将检索 [start_index + cur_idx, end_index + cur_idx] 之间的数据。

        period: int
            用于查询特定时期。
            在 Qlib 中，时期用整数表示（例如 202001 可能代表 2020 年第一季度）
            注意：`period` 会覆盖 `start_index` 和 `end_index`

        返回
        -------
        pd.Series
            索引将是整数，表示数据的时期
            一个典型的例子将是
            TODO

        异常
        ------
        FileNotFoundError
            如果查询的数据不存在，将引发此异常。
        """
        raise NotImplementedError(f"请实现 `period_feature` 方法")


class ExpressionProvider(abc.ABC):
    """表达式提供程序类

    提供表达式数据。
    """

    def __init__(self):
        self.expression_instance_cache = {}

    def get_expression_instance(self, field):
        """获取表达式实例"""
        try:
            if field in self.expression_instance_cache:
                expression = self.expression_instance_cache[field]
            else:
                expression = eval(parse_field(field))
                self.expression_instance_cache[field] = expression
        except NameError as e:
            get_module_logger("data").exception(
                "错误：字段 [%s] 包含无效的运算符/变量 [%s]" % (str(field), str(e).split()[1])
            )
            raise
        except SyntaxError:
            get_module_logger("data").exception("错误：字段 [%s] 包含无效的语法" % str(field))
            raise
        return expression

    @abc.abstractmethod
    def expression(self, instrument, field, start_time=None, end_time=None, freq="day") -> pd.Series:
        """获取表达式数据。

        `expression` 的职责
        - 解析 `field` 并 `load` 相应的数据。
        - 加载数据时，应处理数据的时间依赖性。`get_expression_instance` 通常在此方法中使用

        参数
        ----------
        instrument : str
            一个特定的金融工具。
        field : str
            特征的特定字段。
        start_time : str
            时间范围的开始。
        end_time : str
            时间范围的结束。
        freq : str
            时间频率，可用值：year/quarter/month/week/day。

        返回
        -------
        pd.Series
            特定表达式的数据

            数据有两种格式

            1) 带日期时间索引的表达式

            2) 带整数索引的表达式

                - 因为日期时间不如整数索引高效
        """
        raise NotImplementedError("ExpressionProvider 的子类必须实现 `Expression` 方法")


class DatasetProvider(abc.ABC):
    """数据集提供程序类

    提供数据集数据。
    """

    @abc.abstractmethod
    def dataset(self, instruments, fields, start_time=None, end_time=None, freq="day", inst_processors=[]):
        """获取数据集数据。

        参数
        ----------
        instruments : list or dict
            金融工具列表/字典或股票池配置字典。
        fields : list
            特征实例列表。
        start_time : str
            时间范围的开始。
        end_time : str
            时间范围的结束。
        freq : str
            时间频率。
        inst_processors:  Iterable[Union[dict, InstProcessor]]
            对每个金融工具执行的操作

        返回
        ----------
        pd.DataFrame
            一个带有 <instrument, datetime> 索引的 pandas DataFrame。
        """
        raise NotImplementedError("DatasetProvider 的子类必须实现 `Dataset` 方法")

    def _uri(
        self,
        instruments,
        fields,
        start_time=None,
        end_time=None,
        freq="day",
        disk_cache=1,
        inst_processors=[],
        **kwargs,
    ):
        """获取任务 URI，用于在 qlib_server 中生成 rabbitmq 任务

        参数
        ----------
        instruments : list or dict
            金融工具列表/字典或股票池配置字典。
        fields : list
            特征实例列表。
        start_time : str
            时间范围的开始。
        end_time : str
            时间范围的结束。
        freq : str
            时间频率。
        disk_cache : int
            是否跳过(0)/使用(1)/替换(2)磁盘缓存。

        """
        # TODO: qlib-server 支持 inst_processors
        return DiskDatasetCache._uri(instruments, fields, start_time, end_time, freq, disk_cache, inst_processors)

    @staticmethod
    def get_instruments_d(instruments, freq):
        """
        将不同类型的输入 instruments 解析为输出 instruments_d
        错误的输入 instruments 格式将导致异常。

        """
        if isinstance(instruments, dict):
            if "market" in instruments:
                # 股票池配置字典
                instruments_d = Inst.list_instruments(instruments=instruments, freq=freq, as_list=False)
            else:
                # 金融工具和时间戳字典
                instruments_d = instruments
        elif isinstance(instruments, (list, tuple, pd.Index, np.ndarray)):
            # 一组金融工具的列表或元组
            instruments_d = list(instruments)
        else:
            raise ValueError("参数 `instrument` 的输入类型不支持")
        return instruments_d

    @staticmethod
    def get_column_names(fields):
        """
        从输入字段获取列名

        """
        if len(fields) == 0:
            raise ValueError("字段不能为空")
        column_names = [str(f) for f in fields]
        return column_names

    @staticmethod
    def parse_fields(fields):
        # 解析并检查输入字段
        return [ExpressionD.get_expression_instance(f) for f in fields]

    @staticmethod
    def dataset_processor(instruments_d, column_names, start_time, end_time, freq, inst_processors=[]):
        """
        加载和处理数据，返回数据集。
        - 默认使用多核方法。

        """
        normalize_column_names = normalize_cache_fields(column_names)
        # 每个任务一个进程，以便更快地释放内存。
        workers = max(min(C.get_kernels(freq), len(instruments_d)), 1)

        # 创建迭代器
        if isinstance(instruments_d, dict):
            it = instruments_d.items()
        else:
            it = zip(instruments_d, [None] * len(instruments_d))

        inst_l = []
        task_l = []
        for inst, spans in it:
            inst_l.append(inst)
            task_l.append(
                delayed(DatasetProvider.inst_calculator)(
                    inst, start_time, end_time, freq, normalize_column_names, spans, C, inst_processors
                )
            )

        data = dict(
            zip(
                inst_l,
                ParallelExt(n_jobs=workers, backend=C.joblib_backend, maxtasksperchild=C.maxtasksperchild)(task_l),
            )
        )

        new_data = dict()
        for inst in sorted(data.keys()):
            if len(data[inst]) > 0:
                # 注意：Python 版本 >= 3.6；在 python3.6 之后的版本中，dict 总是保证插入顺序
                new_data[inst] = data[inst]

        if len(new_data) > 0:
            data = pd.concat(new_data, names=["instrument"], sort=False)
            data = DiskDatasetCache.cache_to_origin_data(data, column_names)
        else:
            data = pd.DataFrame(
                index=pd.MultiIndex.from_arrays([[], []], names=("instrument", "datetime")),
                columns=column_names,
                dtype=np.float32,
            )

        return data

    @staticmethod
    def inst_calculator(inst, start_time, end_time, freq, column_names, spans=None, g_config=None, inst_processors=[]):
        """
        计算**一个**金融工具的表达式，返回一个 df 结果。
        如果表达式之前已经计算过，则从缓存加载。

        返回值：一个索引为 'datetime' 和其他数据列的数据帧。

        """
        # FIXME: Windows OS or MacOS using spawn: https://docs.python.org/3.8/library/multiprocessing.html?highlight=spawn#contexts-and-start-methods
        # 注意：此处与 windows 兼容，windows 多进程是 spawn
        C.register_from_C(g_config)

        obj = dict()
        for field in column_names:
            #  客户端没有表达式提供者，数据将使用静态方法从缓存加载。
            obj[field] = ExpressionD.expression(inst, field, start_time, end_time, freq)

        data = pd.DataFrame(obj)
        if not data.empty and not np.issubdtype(data.index.dtype, np.dtype("M")):
            # 如果底层提供的数据不是日期时间格式，我们将其转换为日期时间格式
            _calendar = Cal.calendar(freq=freq)
            data.index = _calendar[data.index.values.astype(int)]
        data.index.names = ["datetime"]

        if not data.empty and spans is not None:
            mask = np.zeros(len(data), dtype=bool)
            for begin, end in spans:
                mask |= (data.index >= begin) & (data.index <= end)
            data = data[mask]

        for _processor in inst_processors:
            if _processor:
                _processor_obj = init_instance_by_config(_processor, accept_types=InstProcessor)
                data = _processor_obj(data, instrument=inst)
        return data


class LocalCalendarProvider(CalendarProvider, ProviderBackendMixin):
    """本地日历数据提供程序类

    从本地数据源提供日历数据。
    """

    def __init__(self, remote=False, backend={}):
        super().__init__()
        self.remote = remote
        self.backend = backend

    def load_calendar(self, freq, future):
        """从文件加载原始日历时间戳。

        参数
        ----------
        freq : str
            读取日历文件的频率。
        future: bool
        返回
        ----------
        list
            时间戳列表
        """
        try:
            backend_obj = self.backend_obj(freq=freq, future=future).data
        except ValueError:
            if future:
                get_module_logger("data").warning(
                    f"加载日历时出错：freq={freq}, future={future}; 返回当前日历！"
                )
                get_module_logger("data").warning(
                    "你可以通过参考以下文档获取未来日历：https://github.com/microsoft/qlib/blob/main/scripts/data_collector/contrib/README.md"
                )
                backend_obj = self.backend_obj(freq=freq, future=False).data
            else:
                raise

        return [pd.Timestamp(x) for x in backend_obj]


class LocalInstrumentProvider(InstrumentProvider, ProviderBackendMixin):
    """本地金融工具数据提供程序类

    从本地数据源提供金融工具数据。
    """

    def __init__(self, backend={}) -> None:
        super().__init__()
        self.backend = backend

    def _load_instruments(self, market, freq):
        return self.backend_obj(market=market, freq=freq).data

    def list_instruments(self, instruments, start_time=None, end_time=None, freq="day", as_list=False):
        market = instruments["market"]
        if market in H["i"]:
            _instruments = H["i"][market]
        else:
            _instruments = self._load_instruments(market, freq=freq)
            H["i"][market] = _instruments
        # 裁剪
        # 使用日历边界
        cal = Cal.calendar(freq=freq)
        start_time = pd.Timestamp(start_time or cal[0])
        end_time = pd.Timestamp(end_time or cal[-1])
        _instruments_filtered = {
            inst: list(
                filter(
                    lambda x: x[0] <= x[1],
                    [(max(start_time, pd.Timestamp(x[0])), min(end_time, pd.Timestamp(x[1]))) for x in spans],
                )
            )
            for inst, spans in _instruments.items()
        }
        _instruments_filtered = {key: value for key, value in _instruments_filtered.items() if value}
        # 过滤
        filter_pipe = instruments["filter_pipe"]
        for filter_config in filter_pipe:
            from . import filter as F  # pylint: disable=C0415

            filter_t = getattr(F, filter_config["filter_type"]).from_config(filter_config)
            _instruments_filtered = filter_t(_instruments_filtered, start_time, end_time, freq)
        # as list
        if as_list:
            return list(_instruments_filtered)
        return _instruments_filtered


class LocalFeatureProvider(FeatureProvider, ProviderBackendMixin):
    """本地特征数据提供程序类

    从本地数据源提供特征数据。
    """

    def __init__(self, remote=False, backend={}):
        super().__init__()
        self.remote = remote
        self.backend = backend

    def feature(self, instrument, field, start_index, end_index, freq):
        # 验证
        field = str(field)[1:]
        instrument = code_to_fname(instrument)
        return self.backend_obj(instrument=instrument, field=field, freq=freq)[start_index : end_index + 1]


class LocalPITProvider(PITProvider):
    # TODO: 添加 PIT 后端文件存储
    # 注意：此类不是多线程安全的！！！！

    def period_feature(self, instrument, field, start_index, end_index, cur_time, period=None):
        if not isinstance(cur_time, pd.Timestamp):
            raise ValueError(
                f"期望 `cur_time` 为 pd.Timestamp，但得到 '{cur_time}'。建议：你不能直接查询 PIT 数据（例如 '$$roewa_q'），必须使用 `P` 运算符将数据转换为每日数据（例如 'P($$roewa_q)')"
            )

        assert end_index <= 0  # PIT 不支持查询未来数据

        DATA_RECORDS = [
            ("date", C.pit_record_type["date"]),
            ("period", C.pit_record_type["period"]),
            ("value", C.pit_record_type["value"]),
            ("_next", C.pit_record_type["index"]),
        ]
        VALUE_DTYPE = C.pit_record_type["value"]

        field = str(field).lower()[2:]
        instrument = code_to_fname(instrument)

        if not field.endswith("_q") and not field.endswith("_a"):
            raise ValueError("时期字段必须以 '_q' 或 '_a' 结尾")
        quarterly = field.endswith("_q")
        index_path = C.dpm.get_data_uri() / "financial" / instrument.lower() / f"{field}.index"
        data_path = C.dpm.get_data_uri() / "financial" / instrument.lower() / f"{field}.data"
        if not (index_path.exists() and data_path.exists()):
            raise FileNotFoundError("未找到文件。")
        data = np.fromfile(data_path, dtype=DATA_RECORDS)

        # 查找 `cur_time` 之前的所有修订时期
        cur_time_int = int(cur_time.year) * 10000 + int(cur_time.month) * 100 + int(cur_time.day)
        loc = np.searchsorted(data["date"], cur_time_int, side="right")
        if loc <= 0:
            return pd.Series(dtype=C.pit_record_type["value"])
        last_period = data["period"][:loc].max()  # 返回最近的季度
        first_period = data["period"][:loc].min()
        period_list = get_period_list(first_period, last_period, quarterly)
        if period is not None:
            # 注意：`period` 的优先级高于 `start_index` 和 `end_index`
            if period not in period_list:
                return pd.Series(dtype=C.pit_record_type["value"])
            else:
                period_list = [period]
        else:
            period_list = period_list[max(0, len(period_list) + start_index - 1) : len(period_list) + end_index]
        value = np.full((len(period_list),), np.nan, dtype=VALUE_DTYPE)
        for i, p in enumerate(period_list):
            value[i], now_period_index = read_period_data(
                index_path, data_path, p, cur_time_int, quarterly
            )
        series = pd.Series(value, index=period_list, dtype=VALUE_DTYPE)

        return series


class LocalExpressionProvider(ExpressionProvider):
    """本地表达式数据提供程序类

    从本地数据源提供表达式数据。
    """

    def __init__(self, time2idx=True):
        super().__init__()
        self.time2idx = time2idx

    def expression(self, instrument, field, start_time=None, end_time=None, freq="day"):
        expression = self.get_expression_instance(field)
        start_time = time_to_slc_point(start_time)
        end_time = time_to_slc_point(end_time)

        # 支持两种查询方式
        # - 基于索引的表达式：这可以节省大量内存，因为日期时间索引不保存在磁盘上
        # - 带日期时间索引数据的表达式：这将使其更方便地与一些现有数据库集成
        if self.time2idx:
            _, _, start_index, end_index = Cal.locate_index(start_time, end_time, freq=freq, future=False)
            lft_etd, rght_etd = expression.get_extended_window_size()
            query_start, query_end = max(0, start_index - lft_etd), end_index + rght_etd
        else:
            start_index, end_index = query_start, query_end = start_time, end_time

        try:
            series = expression.load(instrument, query_start, query_end, freq)
        except Exception as e:
            get_module_logger("data").debug(
                f"加载表达式时出错: "
                f"instrument={instrument}, field=({field}), start_time={start_time}, end_time={end_time}, freq={freq}. "
                f"错误信息: {str(e)}"
            )
            raise
        # 确保每列类型一致
        # FIXME:
        # 1) 当前股票数据是浮点型。如果存在其他类型的数据，这部分需要重新实现。
        # 2) 精度应可配置
        try:
            series = series.astype(np.float32)
        except ValueError:
            pass
        except TypeError:
            pass
        if not series.empty:
            series = series.loc[start_index:end_index]
        return series


class LocalDatasetProvider(DatasetProvider):
    """本地数据集数据提供程序类

    从本地数据源提供数据集数据。
    """

    def __init__(self, align_time: bool = True):
        """
        参数
        ----------
        align_time : bool
            是否将时间对齐到日历
            在某些数据集中，频率是灵活的，无法对齐。
            对于具有共享日历的固定频率数据，将数据对齐到日历将提供以下好处

            - 将查询对齐到相同的参数，以便可以共享缓存。
        """
        super().__init__()
        self.align_time = align_time

    def dataset(
        self,
        instruments,
        fields,
        start_time=None,
        end_time=None,
        freq="day",
        inst_processors=[],
    ):
        instruments_d = self.get_instruments_d(instruments, freq)
        column_names = self.get_column_names(fields)
        if self.align_time:
            # 注意：如果频率是固定值。
            # 将数据对齐到固定的日历点
            cal = Cal.calendar(start_time, end_time, freq)
            if len(cal) == 0:
                return pd.DataFrame(
                    index=pd.MultiIndex.from_arrays([[], []], names=("instrument", "datetime")), columns=column_names
                )
            start_time = cal[0]
            end_time = cal[-1]
        data = self.dataset_processor(
            instruments_d, column_names, start_time, end_time, freq, inst_processors=inst_processors
        )

        return data

    @staticmethod
    def multi_cache_walker(instruments, fields, start_time=None, end_time=None, freq="day"):
        """
        此方法用于为客户端准备表达式缓存。
        然后客户端将自行从表达式缓存加载数据。

        """
        instruments_d = DatasetProvider.get_instruments_d(instruments, freq)
        column_names = DatasetProvider.get_column_names(fields)
        cal = Cal.calendar(start_time, end_time, freq)
        if len(cal) == 0:
            return
        start_time = cal[0]
        end_time = cal[-1]
        workers = max(min(C.kernels, len(instruments_d)), 1)

        ParallelExt(n_jobs=workers, backend=C.joblib_backend, maxtasksperchild=C.maxtasksperchild)(
            delayed(LocalDatasetProvider.cache_walker)(inst, start_time, end_time, freq, column_names)
            for inst in instruments_d
        )

    @staticmethod
    def cache_walker(inst, start_time, end_time, freq, column_names):
        """
        如果一个金融工具的表达式之前没有被计算过，
        计算它并将其写入表达式缓存。

        """
        for field in column_names:
            ExpressionD.expression(inst, field, start_time, end_time, freq)


class ClientCalendarProvider(CalendarProvider):
    """客户端日历数据提供程序类

    作为客户端通过从服务器请求数据来提供日历数据。
    """

    def __init__(self):
        self.conn = None
        self.queue = queue.Queue()

    def set_conn(self, conn):
        self.conn = conn

    def calendar(self, start_time=None, end_time=None, freq="day", future=False):
        self.conn.send_request(
            request_type="calendar",
            request_content={"start_time": str(start_time), "end_time": str(end_time), "freq": freq, "future": future},
            msg_queue=self.queue,
            msg_proc_func=lambda response_content: [pd.Timestamp(c) for c in response_content],
        )
        result = self.queue.get(timeout=C["timeout"])
        return result


class ClientInstrumentProvider(InstrumentProvider):
    """客户端金融工具数据提供程序类

    作为客户端通过从服务器请求数据来提供金融工具数据。
    """

    def __init__(self):
        self.conn = None
        self.queue = queue.Queue()

    def set_conn(self, conn):
        self.conn = conn

    def list_instruments(self, instruments, start_time=None, end_time=None, freq="day", as_list=False):
        def inst_msg_proc_func(response_content):
            if isinstance(response_content, dict):
                instrument = {
                    i: [(pd.Timestamp(s), pd.Timestamp(e)) for s, e in t] for i, t in response_content.items()
                }
            else:
                instrument = response_content
            return instrument

        self.conn.send_request(
            request_type="instrument",
            request_content={
                "instruments": instruments,
                "start_time": str(start_time),
                "end_time": str(end_time),
                "freq": freq,
                "as_list": as_list,
            },
            msg_queue=self.queue,
            msg_proc_func=inst_msg_proc_func,
        )
        result = self.queue.get(timeout=C["timeout"])
        if isinstance(result, Exception):
            raise result
        get_module_logger("data").debug("获取结果")
        return result


class ClientDatasetProvider(DatasetProvider):
    """客户端数据集数据提供程序类

    作为客户端通过从服务器请求数据来提供数据集数据。
    """

    def __init__(self):
        self.conn = None

    def set_conn(self, conn):
        self.conn = conn
        self.queue = queue.Queue()

    def dataset(
        self,
        instruments,
        fields,
        start_time=None,
        end_time=None,
        freq="day",
        disk_cache=0,
        return_uri=False,
        inst_processors=[],
    ):
        if Inst.get_inst_type(instruments) == Inst.DICT:
            get_module_logger("data").warning(
                "不建议从金融工具字典中获取特征，因为特征不会被缓存！"
                "金融工具字典每天都会被清理。"
            )

        if disk_cache == 0:
            """
            调用服务器生成表达式缓存。
            然后直接从表达式缓存加载数据。
            - 默认使用多核方法。

            """
            self.conn.send_request(
                request_type="feature",
                request_content={
                    "instruments": instruments,
                    "fields": fields,
                    "start_time": start_time,
                    "end_time": end_time,
                    "freq": freq,
                    "disk_cache": 0,
                },
                msg_queue=self.queue,
            )
            feature_uri = self.queue.get(timeout=C["timeout"])
            if isinstance(feature_uri, Exception):
                raise feature_uri
            else:
                instruments_d = self.get_instruments_d(instruments, freq)
                column_names = self.get_column_names(fields)
                cal = Cal.calendar(start_time, end_time, freq)
                if len(cal) == 0:
                    return pd.DataFrame(
                        index=pd.MultiIndex.from_arrays([[], []], names=("instrument", "datetime")),
                        columns=column_names,
                    )
                start_time = cal[0]
                end_time = cal[-1]

                data = self.dataset_processor(instruments_d, column_names, start_time, end_time, freq, inst_processors)
                if return_uri:
                    return data, feature_uri
                else:
                    return data
        else:
            """
            调用服务器生成数据集缓存，获取缓存文件的 URI。
            然后直接从 NFS 上的文件加载数据。
            - 使用单进程实现。

            """
            # TODO: 支持 inst_processors, 需要同时修改 qlib-server 的代码
            # FIXME: 重采样后的缓存在再次读取并使用 end_time 截取时，会导致数据日期不完整
            if inst_processors:
                raise ValueError(
                    f"{self.__class__.__name__} 不支持 inst_processor。 "
                    f"请使用 `D.features(disk_cache=0)` 或 `qlib.init(dataset_cache=None)`"
                )
            self.conn.send_request(
                request_type="feature",
                request_content={
                    "instruments": instruments,
                    "fields": fields,
                    "start_time": start_time,
                    "end_time": end_time,
                    "freq": freq,
                    "disk_cache": 1,
                },
                msg_queue=self.queue,
            )
            # - 在回调中完成
            feature_uri = self.queue.get(timeout=C["timeout"])
            if isinstance(feature_uri, Exception):
                raise feature_uri
            get_module_logger("data").debug("获取结果")
            try:
                # 预挂载 nfs, 用于演示
                mnt_feature_uri = C.dpm.get_data_uri(freq).joinpath(C.dataset_cache_dir_name, feature_uri)
                df = DiskDatasetCache.read_data_from_cache(mnt_feature_uri, start_time, end_time, fields)
                get_module_logger("data").debug("完成数据切片")
                if return_uri:
                    return df, feature_uri
                return df
            except AttributeError as attribute_e:
                raise IOError("无法从远程服务器获取金融工具！") from attribute_e


class BaseProvider:
    """本地提供程序类
    它是一组允许用户访问数据的接口。
    由于 PITD 不对用户公开，因此不包含在接口中。

    为了与旧的 qlib 提供程序兼容。
    """

    def calendar(self, start_time=None, end_time=None, freq="day", future=False):
        return Cal.calendar(start_time, end_time, freq, future=future)

    def instruments(self, market="all", filter_pipe=None, start_time=None, end_time=None):
        if start_time is not None or end_time is not None:
            get_module_logger("Provider").warning(
                "金融工具对应于一个股票池。"
                "参数 `start_time` 和 `end_time` 现在不起作用。"
            )
        return InstrumentProvider.instruments(market, filter_pipe)

    def list_instruments(self, instruments, start_time=None, end_time=None, freq="day", as_list=False):
        return Inst.list_instruments(instruments, start_time, end_time, freq, as_list)

    def features(
        self,
        instruments,
        fields,
        start_time=None,
        end_time=None,
        freq="day",
        disk_cache=None,
        inst_processors=[],
    ):
        """
        参数
        ----------
        disk_cache : int
            是否跳过(0)/使用(1)/替换(2)磁盘缓存


        此函数将尝试使用带有关键字 `disk_cache` 的缓存方法，
        如果因为 DatasetD 实例是提供者类而引发类型错误，则将使用提供者方法。
        """
        disk_cache = C.default_disk_cache if disk_cache is None else disk_cache
        fields = list(fields)  # 以防是元组。
        try:
            return DatasetD.dataset(
                instruments, fields, start_time, end_time, freq, disk_cache, inst_processors=inst_processors
            )
        except TypeError:
            return DatasetD.dataset(instruments, fields, start_time, end_time, freq, inst_processors=inst_processors)


class LocalProvider(BaseProvider):
    def _uri(self, type, **kwargs):
        """_uri
        服务器希望获取请求的 URI。URI 将由数据提供者决定。
        例如，不同的缓存层有不同的 URI。

        :param type: URI 的资源类型
        :param **kwargs:
        """
        if type == "calendar":
            return Cal._uri(**kwargs)
        elif type == "instrument":
            return Inst._uri(**kwargs)
        elif type == "feature":
            return DatasetD._uri(**kwargs)

    def features_uri(self, instruments, fields, start_time, end_time, freq, disk_cache=1):
        """features_uri

        返回生成的特征/数据集缓存的 URI

        :param disk_cache:
        :param instruments:
        :param fields:
        :param start_time:
        :param end_time:
        :param freq:
        """
        return DatasetD._dataset_uri(instruments, fields, start_time, end_time, freq, disk_cache)


class ClientProvider(BaseProvider):
    """客户端提供程序

    作为客户端从服务器请求数据。可以发出以下请求：

        - 日历：直接响应日历列表
        - 金融工具（无过滤器）：直接响应金融工具列表/字典
        - 金融工具（有过滤器）：响应金融工具列表/字典
        - 特征：响应缓存URI

    一般工作流程如下：
    当用户使用客户端提供程序发出请求时，客户端提供程序将连接服务器并发送请求。客户端将开始等待响应。响应将立即返回，指示缓存是否可用。只有当客户端收到 `feature_available` 为 true 的响应时，等待过程才会终止。
    `BUG`：每次我们请求特定数据时，都需要连接到服务器，等待响应，然后断开连接。我们无法在单个连接中发出一系列请求。有关 python-socketIO 客户端的文档，请参阅 https://python-socketio.readthedocs.io/en/latest/client.html。
    """

    def __init__(self):
        def is_instance_of_provider(instance: object, cls: type):
            if isinstance(instance, Wrapper):
                p = getattr(instance, "_provider", None)

                return False if p is None else isinstance(p, cls)

            return isinstance(instance, cls)

        from .client import Client  # pylint: disable=C0415

        self.client = Client(C.flask_server, C.flask_port)
        self.logger = get_module_logger(self.__class__.__name__)
        if is_instance_of_provider(Cal, ClientCalendarProvider):
            Cal.set_conn(self.client)
        if is_instance_of_provider(Inst, ClientInstrumentProvider):
            Inst.set_conn(self.client)
        if hasattr(DatasetD, "provider"):
            DatasetD.provider.set_conn(self.client)
        else:
            DatasetD.set_conn(self.client)


import sys

if sys.version_info >= (3, 9):
    from typing import Annotated

    CalendarProviderWrapper = Annotated[CalendarProvider, Wrapper]
    InstrumentProviderWrapper = Annotated[InstrumentProvider, Wrapper]
    FeatureProviderWrapper = Annotated[FeatureProvider, Wrapper]
    PITProviderWrapper = Annotated[PITProvider, Wrapper]
    ExpressionProviderWrapper = Annotated[ExpressionProvider, Wrapper]
    DatasetProviderWrapper = Annotated[DatasetProvider, Wrapper]
    BaseProviderWrapper = Annotated[BaseProvider, Wrapper]
else:
    CalendarProviderWrapper = CalendarProvider
    InstrumentProviderWrapper = InstrumentProvider
    FeatureProviderWrapper = FeatureProvider
    PITProviderWrapper = PITProvider
    ExpressionProviderWrapper = ExpressionProvider
    DatasetProviderWrapper = DatasetProvider
    BaseProviderWrapper = BaseProvider

# 全局数据访问入口
Cal: CalendarProviderWrapper = Wrapper()
Inst: InstrumentProviderWrapper = Wrapper()
FeatureD: FeatureProviderWrapper = Wrapper()
PITD: PITProviderWrapper = Wrapper()
ExpressionD: ExpressionProviderWrapper = Wrapper()
DatasetD: DatasetProviderWrapper = Wrapper()
D: BaseProviderWrapper = Wrapper()


def register_all_wrappers(C):
    """注册所有的包装器"""
    logger = get_module_logger("data")
    module = get_module_by_module_path("qlib.data")

    _calendar_provider = init_instance_by_config(C.calendar_provider, module)
    if getattr(C, "calendar_cache", None) is not None:
        _calendar_provider = init_instance_by_config(C.calendar_cache, module, provide=_calendar_provider)
    register_wrapper(Cal, _calendar_provider, "qlib.data")
    logger.debug(f"注册 Cal {C.calendar_provider}-{C.calendar_cache}")

    _instrument_provider = init_instance_by_config(C.instrument_provider, module)
    register_wrapper(Inst, _instrument_provider, "qlib.data")
    logger.debug(f"注册 Inst {C.instrument_provider}")

    if getattr(C, "feature_provider", None) is not None:
        feature_provider = init_instance_by_config(C.feature_provider, module)
        register_wrapper(FeatureD, feature_provider, "qlib.data")
        logger.debug(f"注册 FeatureD {C.feature_provider}")

    if getattr(C, "pit_provider", None) is not None:
        pit_provider = init_instance_by_config(C.pit_provider, module)
        register_wrapper(PITD, pit_provider, "qlib.data")
        logger.debug(f"注册 PITD {C.pit_provider}")

    if getattr(C, "expression_provider", None) is not None:
        # 在客户端提供者中，此提供者是不必要的
        _eprovider = init_instance_by_config(C.expression_provider, module)
        if getattr(C, "expression_cache", None) is not None:
            _eprovider = init_instance_by_config(C.expression_cache, module, provider=_eprovider)
        register_wrapper(ExpressionD, _eprovider, "qlib.data")
        logger.debug(f"注册 ExpressionD {C.expression_provider}-{C.expression_cache}")

    _dprovider = init_instance_by_config(C.dataset_provider, module)
    if getattr(C, "dataset_cache", None) is not None:
        _dprovider = init_instance_by_config(C.dataset_cache, module, provider=_dprovider)
    register_wrapper(DatasetD, _dprovider, "qlib.data")
    logger.debug(f"注册 DatasetD {C.dataset_provider}-{C.dataset_cache}")

    register_wrapper(D, C.provider, "qlib.data")
    logger.debug(f"注册 D {C.provider}")
