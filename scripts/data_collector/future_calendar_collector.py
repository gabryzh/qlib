# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

# 导入abc模块，用于定义抽象基类
import abc
# 导入importlib模块，用于动态导入模块
import importlib
# 从pathlib导入Path类，用于处理文件路径
from pathlib import Path
# 从typing导入Union, Iterable, List，用于类型注解
from typing import Union, Iterable, List

# 导入fire库，用于创建命令行界面
import fire
# 导入numpy库，用于数值操作
import numpy as np
# 导入pandas库，用于数据处理
import pandas as pd

# 导入baostock库，用于获取A股历史数据
# 需要通过 `pip install baostock` 安装
import baostock as bs
# 导入loguru库，用于日志记录
from loguru import logger


class CollectorFutureCalendar:
    """
    收集未来交易日历的基类。
    """
    calendar_format = "%Y-%m-%d"

    def __init__(self, qlib_dir: Union[str, Path], start_date: str = None, end_date: str = None):
        """
        初始化。

        Parameters
        ----------
        qlib_dir: str or Path
            qlib数据目录的路径。
        start_date: str
            开始日期。如果为None，则从现有日历的最后一天开始。
        end_date: str
            结束日期。如果为None，则默认为现有日历最后一天之后的两年。
        """
        self.qlib_dir = Path(qlib_dir).expanduser().absolute()
        # 现有日历文件的路径
        self.calendar_path = self.qlib_dir.joinpath("calendars/day.txt")
        # 将要保存的未来日历文件的路径
        self.future_path = self.qlib_dir.joinpath("calendars/day_future.txt")

        self._calendar_list = self.calendar_list
        _latest_date = self._calendar_list[-1]

        # 确定获取未来日历的起止时间
        self.start_date = _latest_date if start_date is None else pd.Timestamp(start_date)
        self.end_date = _latest_date + pd.Timedelta(days=365 * 2) if end_date is None else pd.Timestamp(end_date)

    @property
    def calendar_list(self) -> List[pd.Timestamp]:
        """加载并返回现有的交易日历列表。"""
        if not self.calendar_path.exists():
            raise ValueError(f"日历文件不存在: {self.calendar_path}")

        calendar_df = pd.read_csv(self.calendar_path, header=None)
        calendar_df.columns = ["date"]
        calendar_df["date"] = pd.to_datetime(calendar_df["date"])
        return calendar_df["date"].to_list()

    def _format_datetime(self, datetime_d: [str, pd.Timestamp]):
        """将日期时间对象格式化为字符串。"""
        datetime_d = pd.Timestamp(datetime_d)
        return datetime_d.strftime(self.calendar_format)

    def write_calendar(self, calendar: Iterable):
        """将合并后的日历（现有+未来）写入文件。"""
        # 合并、去重、排序
        calendars_list = [self._format_datetime(x) for x in sorted(set(self.calendar_list + list(calendar)))]
        # 写入文件
        np.savetxt(self.future_path, calendars_list, fmt="%s", encoding="utf-8")

    @abc.abstractmethod
    def collector(self) -> Iterable[pd.Timestamp]:
        """
        抽象方法：收集交易日历。
        子类必须实现此方法以提供特定市场（如中国、美国）的日历获取逻辑。
        """
        raise NotImplementedError(f"请实现 `collector` 方法")


class CollectorFutureCalendarCN(CollectorFutureCalendar):
    """
    收集中国A股未来交易日历的实现类。
    """
    def collector(self) -> Iterable[pd.Timestamp]:
        """使用baostock库获取中国交易日历。"""
        # 登录baostock
        lg = bs.login()
        if lg.error_code != "0":
            raise ValueError(f"baostock登录失败: {lg.error_msg}")

        # 查询交易日数据
        rs = bs.query_trade_dates(
            start_date=self._format_datetime(self.start_date), end_date=self._format_datetime(self.end_date)
        )
        if rs.error_code != "0":
            raise ValueError(f"查询交易日失败: {rs.error_msg}")

        # 解析返回的数据
        data_list = []
        while (rs.error_code == "0") & rs.next():
            data_list.append(rs.get_row_data())

        # 登出baostock
        bs.logout()

        if not data_list:
            return []

        calendar = pd.DataFrame(data_list, columns=rs.fields)
        calendar["is_trading_day"] = calendar["is_trading_day"].astype(int)
        # 筛选出is_trading_day为1的日期
        trading_days = pd.to_datetime(calendar[calendar["is_trading_day"] == 1]["calendar_date"]).to_list()
        return trading_days


class CollectorFutureCalendarUS(CollectorFutureCalendar):
    """
    收集美国股市未来交易日历的实现类（占位符）。
    """
    def collector(self) -> Iterable[pd.Timestamp]:
        # TODO: 实现美国未来日历的获取逻辑
        raise ValueError("尚不支持美国日历")


def run(qlib_dir: Union[str, Path], region: str = "cn", start_date: str = None, end_date: str = None):
    """
    运行未来交易日历收集的主函数。

    Parameters
    ----------
    qlib_dir: str or Path
        qlib数据目录。
    region: str
        区域，支持 'cn'/'CN' 或 'us'/'US'。
    start_date: str
        开始日期。
    end_date: str
        结束日期。

    Examples
    -------
        # 获取中国未来交易日历
        $ python future_calendar_collector.py run --qlib_dir <你的数据目录> --region cn
    """
    logger.info(f"开始收集未来交易日历: 区域={region}")
    # 根据区域动态选择对应的Collector类
    _cur_module = importlib.import_module("future_calendar_collector")
    _class_name = f"CollectorFutureCalendar{region.upper()}"
    if hasattr(_cur_module, _class_name):
        _class = getattr(_cur_module, _class_name)
        # 实例化并运行
        collector = _class(qlib_dir=qlib_dir, start_date=start_date, end_date=end_date)
        future_calendar = collector.collector()
        collector.write_calendar(future_calendar)
        logger.info("未来交易日历收集完成。")
    else:
        raise ValueError(f"不支持的区域: {region}")


if __name__ == "__main__":
    # 使用fire库将run函数暴露为命令行工具
    fire.Fire(run)
