# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

# 区域常量
from typing import TypeVar

import numpy as np
import pandas as pd

# 区域常量
REG_CN = "cn"  # 中国
REG_US = "us"  # 美国
REG_TW = "tw"  # 台湾

# 用于避免除以零的极小值
EPS = 1e-12

# 整数表示的无穷大
INF = int(1e18)
ONE_DAY = pd.Timedelta("1day")  # 一天
ONE_MIN = pd.Timedelta("1min")  # 一分钟
EPS_T = pd.Timedelta("1s")  # 用于排除右侧区间点的一秒钟
float_or_ndarray = TypeVar("float_or_ndarray", float, np.ndarray)  # 类型变量，可以是浮点数或numpy数组
