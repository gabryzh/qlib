# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.
"""
关于配置
==========

配置将基于 _default_config。
支持两种模式
- 客户端
- 服务器

"""
from __future__ import annotations

import os
import re
import copy
import logging
import platform
import multiprocessing
from pathlib import Path
from typing import Callable, Optional, Union
from typing import TYPE_CHECKING

from qlib.constant import REG_CN, REG_US, REG_TW

if TYPE_CHECKING:
    from qlib.utils.time import Freq

from pydantic_settings import BaseSettings, SettingsConfigDict


class MLflowSettings(BaseSettings):
    uri: str = "file:" + str(Path(os.getcwd()).resolve() / "mlruns")
    default_exp_name: str = "Experiment"


class QSettings(BaseSettings):
    """
    Qlib的设置。
    它试图为Qlib的大多数组件提供默认设置。
    但是，为Qlib的所有组件提供全面的设置将是一个漫长的过程。

    以下是一些设计准则：
    - 设置的优先级是
        - 主动传入的设置，例如 `qlib.init(provider_uri=...)`
        - 默认设置
            - QSettings试图为Qlib的大多数组件提供默认设置。
    """

    mlflow: MLflowSettings = MLflowSettings()
    provider_uri: str = "~/.qlib/qlib_data/cn_data"

    model_config = SettingsConfigDict(
        env_prefix="QLIB_",
        env_nested_delimiter="_",
    )


QSETTINGS = QSettings()


class Config:
    """配置类"""
    def __init__(self, default_conf):
        self.__dict__["_default_config"] = copy.deepcopy(default_conf)  # 避免与__getattr__冲突
        self.reset()

    def __getitem__(self, key):
        return self.__dict__["_config"][key]

    def __getattr__(self, attr):
        if attr in self.__dict__["_config"]:
            return self.__dict__["_config"][attr]

        raise AttributeError(f"No such `{attr}` in self._config")

    def get(self, key, default=None):
        return self.__dict__["_config"].get(key, default)

    def __setitem__(self, key, value):
        self.__dict__["_config"][key] = value

    def __setattr__(self, attr, value):
        self.__dict__["_config"][attr] = value

    def __contains__(self, item):
        return item in self.__dict__["_config"]

    def __getstate__(self):
        return self.__dict__

    def __setstate__(self, state):
        self.__dict__.update(state)

    def __str__(self):
        return str(self.__dict__["_config"])

    def __repr__(self):
        return str(self.__dict__["_config"])

    def reset(self):
        self.__dict__["_config"] = copy.deepcopy(self._default_config)

    def update(self, *args, **kwargs):
        self.__dict__["_config"].update(*args, **kwargs)

    def set_conf_from_C(self, config_c):
        self.update(**config_c.__dict__["_config"])

    @staticmethod
    def register_from_C(config, skip_register=True):
        from .utils import set_log_with_config  # pylint: disable=C0415

        if C.registered and skip_register:
            return

        C.set_conf_from_C(config)
        if C.logging_config:
            set_log_with_config(C.logging_config)
        C.register()


# pickle.dump协议版本: https://docs.python.org/3/library/pickle.html#data-stream-format
PROTOCOL_VERSION = 4

NUM_USABLE_CPU = max(multiprocessing.cpu_count() - 2, 1)

DISK_DATASET_CACHE = "DiskDatasetCache"
SIMPLE_DATASET_CACHE = "SimpleDatasetCache"
DISK_EXPRESSION_CACHE = "DiskExpressionCache"

DEPENDENCY_REDIS_CACHE = (DISK_DATASET_CACHE, DISK_EXPRESSION_CACHE)

_default_config = {
    # 数据提供者配置
    "calendar_provider": "LocalCalendarProvider",
    "instrument_provider": "LocalInstrumentProvider",
    "feature_provider": "LocalFeatureProvider",
    "pit_provider": "LocalPITProvider",
    "expression_provider": "LocalExpressionProvider",
    "dataset_provider": "LocalDatasetProvider",
    "provider": "LocalProvider",
    # 在qlib.init()中配置
    # "provider_uri" str or dict:
    #   # str
    #   "~/.qlib/stock_data/cn_data"
    #   # dict
    #   {"day": "~/.qlib/stock_data/cn_data", "1min": "~/.qlib/stock_data/cn_data_1min"}
    # 注意: provider_uri的优先级:
    #   1. backend_config: backend_obj["kwargs"]["provider_uri"]
    #   2. backend_config: backend_obj["kwargs"]["provider_uri_map"]
    #   3. qlib.init: provider_uri
    "provider_uri": "",
    # 缓存
    "expression_cache": None,
    "calendar_cache": None,
    # 简单数据集缓存
    "local_cache_path": None,
    # 内核可以是固定值或可调用函数，如 `def (freq: str) -> int`
    # 如果内核是arctic_kernels，`min(NUM_USABLE_CPU, 30)`可能是一个好值
    "kernels": NUM_USABLE_CPU,
    # pickle.dump协议版本
    "dump_protocol_version": PROTOCOL_VERSION,
    # 一个进程属于多少个任务。建议高频数据为1，日数据为None。
    "maxtasksperchild": None,
    # 如果joblib_backend为None，则使用loky
    "joblib_backend": "multiprocessing",
    "default_disk_cache": 1,  # 0:跳过/1:使用
    "mem_cache_size_limit": 500,
    "mem_cache_limit_type": "length",
    # 内存缓存过期秒数，仅在'DatasetURICache'和'client D.calendar'中使用
    # 默认1小时
    "mem_cache_expire": 60 * 60,
    # 缓存目录名称
    "dataset_cache_dir_name": "dataset_cache",
    "features_cache_dir_name": "features_cache",
    # redis
    # 为了使用缓存
    "redis_host": "127.0.0.1",
    "redis_port": 6379,
    "redis_task_db": 1,
    "redis_password": None,
    # 此值可以通过qlib.init重置
    "logging_level": logging.INFO,
    # qlib日志的全局配置
    # logging_level可以更精细地控制日志级别
    "logging_config": {
        "version": 1,
        "formatters": {
            "logger_format": {
                "format": "[%(process)s:%(threadName)s](%(asctime)s) %(levelname)s - %(name)s - [%(filename)s:%(lineno)d] - %(message)s"
            }
        },
        "filters": {
            "field_not_found": {
                "()": "qlib.log.LogFilter",
                "param": [".*?WARN: data not found for.*?"],
            }
        },
        "handlers": {
            "console": {
                "class": "logging.StreamHandler",
                "level": logging.DEBUG,
                "formatter": "logger_format",
                "filters": ["field_not_found"],
            }
        },
        # 通常这应该设置为 `False` 以避免重复记录 [1]。
        # 但是，由于pytest中的错误，它要求日志消息传播到根记录器以被 `caplog` 捕获 [2]。
        # [1] https://github.com/microsoft/qlib/pull/1661
        # [2] https://github.com/pytest-dev/pytest/issues/3697
        "loggers": {"qlib": {"level": logging.DEBUG, "handlers": ["console"], "propagate": False}},
        # 为了让qlib与其他包一起工作，我们不应该禁用现有的记录器。
        # 注意，根据日志记录的文档，此参数默认为True。
        "disable_existing_loggers": False,
    },
    # 实验管理器的默认配置
    "exp_manager": {
        "class": "MLflowExpManager",
        "module_path": "qlib.workflow.expm",
        "kwargs": {
            "uri": QSETTINGS.mlflow.uri,
            "default_exp_name": QSETTINGS.mlflow.default_exp_name,
        },
    },
    "pit_record_type": {
        "date": "I",  # uint32
        "period": "I",  # uint32
        "value": "d",  # float64
        "index": "I",  # uint32
    },
    "pit_record_nan": {
        "date": 0,
        "period": 0,
        "value": float("NAN"),
        "index": 0xFFFFFFFF,
    },
    # MongoDB的默认配置
    "mongo": {
        "task_url": "mongodb://localhost:27017/",
        "task_db_name": "default_task_db",
    },
    # 高频分钟数据的移位分钟，在回测中使用
    # 如果min_data_shift == 0，使用默认市场时间 [9:30, 11:29, 1:00, 2:59]
    # 如果min_data_shift != 0，使用移位的市场时间 [9:30, 11:29, 1:00, 2:59] - shift*minute
    "min_data_shift": 0,
}

MODE_CONF = {
    "server": {
        # 在qlib.init()中配置
        "provider_uri": "",
        # redis
        "redis_host": "127.0.0.1",
        "redis_port": 6379,
        "redis_task_db": 1,
        # 缓存
        "expression_cache": DISK_EXPRESSION_CACHE,
        "dataset_cache": DISK_DATASET_CACHE,
        "local_cache_path": Path("~/.cache/qlib_simple_cache").expanduser().resolve(),
        "mount_path": None,
    },
    "client": {
        # 在用户自己的代码中配置
        "provider_uri": QSETTINGS.provider_uri,
        # 缓存
        # 使用参数'remote'来声明客户端正在使用服务器缓存，并且写访问将被禁用。
        # 默认禁用缓存。避免为初学者引入高级功能
        "dataset_cache": None,
        # SimpleDatasetCache目录
        "local_cache_path": Path("~/.cache/qlib_simple_cache").expanduser().resolve(),
        # 客户端配置
        "mount_path": None,
        "auto_mount": False,  # nfs已在我们的服务器上挂载[auto_mount: False]。
        # nfs应该由qlib在其他服务器上自动挂载
        # 服务器（例如PAI）[auto_mount:True]
        "timeout": 100,
        "logging_level": logging.INFO,
        "region": REG_CN,
        # 自定义运算符
        # custom_ops的每个元素都应该是Type[ExpressionOps]或dict
        # 如果custom_ops的元素是Type[ExpressionOps]，它表示自定义运算符类
        # 如果custom_ops的元素是dict，它表示自定义运算符的配置，并且应包括`class`和`module_path`键。
        "custom_ops": [],
    },
}

HIGH_FREQ_CONFIG = {
    "provider_uri": "~/.qlib/qlib_data/cn_data_1min",
    "dataset_cache": None,
    "expression_cache": "DiskExpressionCache",
    "region": REG_CN,
}

_default_region_config = {
    REG_CN: {
        "trade_unit": 100,
        "limit_threshold": 0.095,
        "deal_price": "close",
    },
    REG_US: {
        "trade_unit": 1,
        "limit_threshold": None,
        "deal_price": "close",
    },
    REG_TW: {
        "trade_unit": 1000,
        "limit_threshold": 0.1,
        "deal_price": "close",
    },
}


class QlibConfig(Config):
    # URI类型
    LOCAL_URI = "local"
    NFS_URI = "nfs"
    DEFAULT_FREQ = "__DEFAULT_FREQ"

    def __init__(self, default_conf):
        super().__init__(default_conf)
        self._registered = False

    class DataPathManager:
        """
        动机:
        - 根据给定信息（例如provider_uri、mount_path和frequency）获取访问数据的正确路径（例如数据uri）
        - 一些处理uri的辅助函数。
        """

        def __init__(self, provider_uri: Union[str, Path, dict], mount_path: Union[str, Path, dict]):
            """
            `provider_uri`和`mount_path`的关系
            - `mount_path`仅在provider_uri是NFS路径时使用
            - 否则，provider_uri将用于访问数据
            """
            self.provider_uri = provider_uri
            self.mount_path = mount_path

        @staticmethod
        def format_provider_uri(provider_uri: Union[str, dict, Path]) -> dict:
            if provider_uri is None:
                raise ValueError("provider_uri cannot be None")
            if isinstance(provider_uri, (str, dict, Path)):
                if not isinstance(provider_uri, dict):
                    provider_uri = {QlibConfig.DEFAULT_FREQ: provider_uri}
            else:
                raise TypeError(f"provider_uri does not support {type(provider_uri)}")
            for freq, _uri in provider_uri.items():
                if QlibConfig.DataPathManager.get_uri_type(_uri) == QlibConfig.LOCAL_URI:
                    provider_uri[freq] = str(Path(_uri).expanduser().resolve())
            return provider_uri

        @staticmethod
        def get_uri_type(uri: Union[str, Path]):
            uri = uri if isinstance(uri, str) else str(uri.expanduser().resolve())
            is_win = re.match("^[a-zA-Z]:.*", uri) is not None  # such as 'C:\\data', 'D:'
            # such as 'host:/data/'   (User may define short hostname by themselves or use localhost)
            is_nfs_or_win = re.match("^[^/]+:.+", uri) is not None

            if is_nfs_or_win and not is_win:
                return QlibConfig.NFS_URI
            else:
                return QlibConfig.LOCAL_URI

        def get_data_uri(self, freq: Optional[Union[str, Freq]] = None) -> Path:
            """
            请参考DataPathManager的__init__和类文档
            """
            if freq is not None:
                freq = str(freq)  # 将Freq转换为字符串
            if freq is None or freq not in self.provider_uri:
                freq = QlibConfig.DEFAULT_FREQ
            _provider_uri = self.provider_uri[freq]
            if self.get_uri_type(_provider_uri) == QlibConfig.LOCAL_URI:
                return Path(_provider_uri)
            elif self.get_uri_type(_provider_uri) == QlibConfig.NFS_URI:
                if "win" in platform.system().lower():
                    # windows, mount_path是驱动器
                    _path = str(self.mount_path[freq])
                    return Path(f"{_path}:\\") if ":" not in _path else Path(_path)
                return Path(self.mount_path[freq])
            else:
                raise NotImplementedError(f"不支持此类型的uri")

    def set_mode(self, mode):
        # 引发KeyError
        self.update(MODE_CONF[mode])
        # TODO: 根据kwargs更新区域

    def set_region(self, region):
        # 引发KeyError
        self.update(_default_region_config[region])

    @staticmethod
    def is_depend_redis(cache_name: str):
        return cache_name in DEPENDENCY_REDIS_CACHE

    @property
    def dpm(self):
        return self.DataPathManager(self["provider_uri"], self["mount_path"])

    def resolve_path(self):
        # 解析路径
        _mount_path = self["mount_path"]
        _provider_uri = self.DataPathManager.format_provider_uri(self["provider_uri"])
        if not isinstance(_mount_path, dict):
            _mount_path = {_freq: _mount_path for _freq in _provider_uri.keys()}

        # 检查provider_uri和mount_path
        _miss_freq = set(_provider_uri.keys()) - set(_mount_path.keys())
        assert len(_miss_freq) == 0, f"mount_path缺少频率: {_miss_freq}"

        # 解析
        for _freq in _provider_uri.keys():
            # mount_path
            _mount_path[_freq] = (
                _mount_path[_freq] if _mount_path[_freq] is None else str(Path(_mount_path[_freq]).expanduser())
            )
        self["provider_uri"] = _provider_uri
        self["mount_path"] = _mount_path

    def set(self, default_conf: str = "client", **kwargs):
        """
        根据输入参数配置qlib

        配置将像字典一样工作。

        通常，它会根据键逐字替换值。
        但是，当配置嵌套且复杂时，用户有时很难设置配置。

        因此，此API提供了一些特殊参数，供用户以更方便的方式设置键。
        - region: REG_CN, REG_US
            - 几个与区域相关的配置将被更改

        参数
        ----------
        default_conf : str
            用户选择的默认配置模板: "server", "client"
        """
        from .utils import set_log_with_config, get_module_logger, can_use_cache  # pylint: disable=C0415

        self.reset()

        _logging_config = kwargs.get("logging_config", self.logging_config)

        # set global config
        if _logging_config:
            set_log_with_config(_logging_config)

        logger = get_module_logger("Initialization", kwargs.get("logging_level", self.logging_level))
        logger.info(f"default_conf: {default_conf}.")

        self.set_mode(default_conf)
        self.set_region(kwargs.get("region", self["region"] if "region" in self else REG_CN))

        for k, v in kwargs.items():
            if k not in self:
                logger.warning("Unrecognized config %s" % k)
            self[k] = v

        self.resolve_path()

        if not (self["expression_cache"] is None and self["dataset_cache"] is None):
            # 检查redis
            if not can_use_cache():
                log_str = ""
                # 检查表达式缓存
                if self.is_depend_redis(self["expression_cache"]):
                    log_str += self["expression_cache"]
                    self["expression_cache"] = None
                # 检查数据集缓存
                if self.is_depend_redis(self["dataset_cache"]):
                    log_str += f" and {self['dataset_cache']}" if log_str else self["dataset_cache"]
                    self["dataset_cache"] = None
                if log_str:
                    logger.warning(
                        f"redis连接失败(host={self['redis_host']} port={self['redis_port']}), "
                        f"{log_str} 将不被使用！"
                    )

    def register(self):
        from .utils import init_instance_by_config  # pylint: disable=C0415
        from .data.ops import register_all_ops  # pylint: disable=C0415
        from .data.data import register_all_wrappers  # pylint: disable=C0415
        from .workflow import R, QlibRecorder  # pylint: disable=C0415
        from .workflow.utils import experiment_exit_handler  # pylint: disable=C0415

        register_all_ops(self)
        register_all_wrappers(self)
        # 设置QlibRecorder
        exp_manager = init_instance_by_config(self["exp_manager"])
        qr = QlibRecorder(exp_manager)
        R.register(qr)
        # python程序结束时清理实验
        experiment_exit_handler()

        # 支持用户重置qlib版本（当用户希望连接到旧版本的qlib服务器时很有用）
        self.reset_qlib_version()

        self._registered = True

    def reset_qlib_version(self):
        import qlib  # pylint: disable=C0415

        reset_version = self.get("qlib_reset_version", None)
        if reset_version is not None:
            qlib.__version__ = reset_version
        else:
            qlib.__version__ = getattr(qlib, "__version__bak")
            # Due to a bug? that converting __version__ to _QlibConfig__version__bak
            # Using  __version__bak instead of __version__

    def get_kernels(self, freq: str):
        """给定频率获取处理器数量"""
        if isinstance(self["kernels"], Callable):
            return self["kernels"](freq)
        return self["kernels"]

    @property
    def registered(self):
        return self._registered


# 全局配置
C = QlibConfig(_default_config)
