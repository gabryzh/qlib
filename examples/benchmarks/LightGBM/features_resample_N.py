#  Copyright (c) Microsoft Corporation.
#  Licensed under the MIT License.

import pandas as pd

from qlib.data.inst_processor import InstProcessor
from qlib.utils.resam import resam_calendar


class ResampleNProcessor(InstProcessor):
    """
    重采样N处理器。
    """
    def __init__(self, target_frq: str, **kwargs):
        """
        初始化重采样N处理器。

        :param target_frq: 目标频率。
        """
        self.target_frq = target_frq

    def __call__(self, df: pd.DataFrame, *args, **kwargs):
        """
        对数据帧进行重采样。

        :param df: 输入的数据帧。
        :return: 重采样后的数据帧。
        """
        df.index = pd.to_datetime(df.index)
        res_index = resam_calendar(df.index, "1min", self.target_frq)
        df = df.resample(self.target_frq).last().reindex(res_index)
        return df
