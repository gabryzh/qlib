# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

# qlib.data 模块的入口文件
# 用户可以通过 `from qlib.data import D` 的方式快速开始
# 用户也可以根据自己的需求，从 `qlib.data` 导入其他模块

from __future__ import division
from __future__ import print_function

# 从 .data 模块中导入数据相关的类
from .data import (
    D,  # D 是一个便捷的工具，用于获取数据
    CalendarProvider,  # 日历提供者
    InstrumentProvider,  # 股票列表提供者
    FeatureProvider,  # 特征提供者
    ExpressionProvider,  # 表达式提供者
    DatasetProvider,  # 数据集提供者
    LocalCalendarProvider,  # 本地日历提供者
    LocalInstrumentProvider,  # 本地股票列表提供者
    LocalFeatureProvider,  # 本地特征提供者
    LocalPITProvider,  # 本地 Point-in-Time 提供者
    LocalExpressionProvider,  # 本地表达式提供者
    LocalDatasetProvider,  # 本地数据集提供者
    ClientCalendarProvider,  # 客户端日历提供者
    ClientInstrumentProvider,  # 客户端股票列表提供者
    ClientDatasetProvider,  # 客户端数据集提供者
    BaseProvider,  # 提供者的基类
    LocalProvider,  # 本地提供者
    ClientProvider,  # 客户端提供者
)

# 从 .cache 模块中导入缓存相关的类
from .cache import (
    ExpressionCache,  # 表达式缓存
    DatasetCache,  # 数据集缓存
    DiskExpressionCache,  # 磁盘表达式缓存
    DiskDatasetCache,  # 磁盘数据集缓存
    SimpleDatasetCache,  # 简单数据集缓存
    DatasetURICache,  # 数据集 URI 缓存
    MemoryCalendarCache,  # 内存日历缓存
)


# 定义了 `from qlib.data import *` 时会导入的模块
__all__ = [
    "D",
    "CalendarProvider",
    "InstrumentProvider",
    "FeatureProvider",
    "ExpressionProvider",
    "DatasetProvider",
    "LocalCalendarProvider",
    "LocalInstrumentProvider",
    "LocalFeatureProvider",
    "LocalPITProvider",
    "LocalExpressionProvider",
    "LocalDatasetProvider",
    "ClientCalendarProvider",
    "ClientInstrumentProvider",
    "ClientDatasetProvider",
    "BaseProvider",
    "LocalProvider",
    "ClientProvider",
    "ExpressionCache",
    "DatasetCache",
    "DiskExpressionCache",
    "DiskDatasetCache",
    "SimpleDatasetCache",
    "DatasetURICache",
    "MemoryCalendarCache",
]
