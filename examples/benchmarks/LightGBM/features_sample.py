import datetime
import pandas as pd

from qlib.data.inst_processor import InstProcessor


class Resample1minProcessor(InstProcessor):
    """此处理器尝试对数据进行重采样。它将通过选择特定的分钟将数据从 1 分钟频率重采样到日频率"""

    def __init__(self, hour: int, minute: int, **kwargs):
        """
        初始化 Resample1minProcessor。

        :param hour: 要选择的小时。
        :param minute: 要选择的分钟。
        """
        self.hour = hour
        self.minute = minute

    def __call__(self, df: pd.DataFrame, *args, **kwargs):
        """
        对数据帧进行重采样。

        :param df: 输入的数据帧。
        :return: 重采样后的数据帧。
        """
        df.index = pd.to_datetime(df.index)
        df = df.loc[df.index.time == datetime.time(self.hour, self.minute)]
        df.index = df.index.normalize()
        return df
