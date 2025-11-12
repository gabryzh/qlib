import abc
import json
import pandas as pd


class InstProcessor:
    """金融工具处理器基类

    这是一个抽象基类，用于定义处理单个金融工具数据的处理器。
    用户可以继承这个类来创建自定义的数据处理器。
    """
    @abc.abstractmethod
    def __call__(self, df: pd.DataFrame, instrument, *args, **kwargs):
        """
        处理数据

        注意： **处理器可能会就地修改 `df` 的内容 !!!!! **
        用户应该在外部保留一份数据的副本

        Parameters
        ----------
        df : pd.DataFrame
            来自 handler 的原始 DataFrame 或来自上一个处理器的结果。
        instrument: str
            正在处理的金融工具的ID。
        """

    def __str__(self):
        """
        返回处理器的字符串表示形式，方便调试和日志记录。
        """
        return f"{self.__class__.__name__}:{json.dumps(self.__dict__, sort_keys=True, default=str)}"
