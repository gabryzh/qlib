# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

import sys
from typing import List
from pathlib import Path

import fire
import numpy as np
import pandas as pd
from loguru import logger

# 导入baostock库用于获取A股交易日数据
import baostock as bs

# 将上三级目录添加到系统路径，以便导入utils模块
CUR_DIR = Path(__file__).resolve().parent
sys.path.append(str(CUR_DIR.parent.parent.parent))


from data_collector.utils import generate_minutes_calendar_from_daily


def read_calendar_from_qlib(qlib_dir: Path) -> pd.DataFrame:
    """
    从指定的Qlib数据目录中读取现有的日线交易日历。

    Parameters
    ----------
    qlib_dir: Path
        Qlib数据目录的路径。

    Returns
    -------
    pd.DataFrame:
        包含日历日期的DataFrame，如果文件不存在则返回空的DataFrame。
    """
    calendar_path = qlib_dir.joinpath("calendars").joinpath("day.txt")
    if not calendar_path.exists():
        return pd.DataFrame()
    return pd.read_csv(calendar_path, header=None)


def write_calendar_to_qlib(qlib_dir: Path, date_list: List[str], freq: str = "day"):
    """
    将生成的未来交易日历写入到Qlib的calendars目录。

    Parameters
    ----------
    qlib_dir: Path
        Qlib数据目录的路径。
    date_list: List[str]
        要写入的日期列表。
    freq: str
        数据频率, "day" 或 "1min"。文件名将是 `<freq>_future.txt`。
    """
    calendar_path = str(qlib_dir.joinpath("calendars").joinpath(f"{freq}_future.txt"))
    np.savetxt(calendar_path, date_list, fmt="%s", encoding="utf-8")
    logger.info(f"成功写入未来交易日历: {calendar_path}")


def generate_qlib_calendar(date_list: List[str], freq: str) -> List[str]:
    """
    根据给定的日线日历和频率，生成相应频率的日历列表。

    Parameters
    ----------
    date_list: List[str]
        日线交易日历列表。
    freq: str
        目标频率, "day" 或 "1min"。

    Returns
    -------
    List[str]:
        目标频率的日历列表。
    """
    if freq == "day":
        return date_list
    elif freq == "1min":
        # 如果是分钟线，则根据日线日历生成分钟线时间戳
        min_list = generate_minutes_calendar_from_daily(date_list, freq=freq).tolist()
        return [pd.Timestamp(dt).strftime("%Y-%m-%d %H:%M:%S") for dt in min_list]
    else:
        raise ValueError(f"不支持的频率: {freq}")


def future_calendar_collector(qlib_dir: [str, Path], freq: str = "day"):
    """
    收集未来的交易日历并保存到Qlib目录。

    Parameters
    ----------
    qlib_dir: str or Path
        Qlib数据目录的路径。
    freq: str
        目标日历的频率, 可选值为 ["day", "1min"]。
    """
    qlib_dir = Path(qlib_dir).expanduser().resolve()
    if not qlib_dir.exists():
        raise FileNotFoundError(f"指定的Qlib目录不存在: {qlib_dir}")

    # 登录Baostock
    lg = bs.login()
    if lg.error_code != "0":
        logger.error(f"Baostock登录失败: {lg.error_msg}")
        return

    # 读取Qlib中已有的日历
    daily_calendar = read_calendar_from_qlib(qlib_dir)
    end_year = pd.Timestamp.now().year

    # 确定查询的开始年份
    if daily_calendar.empty:
        # 如果没有历史日历，从当年开始
        start_year = pd.Timestamp.now().year
    else:
        # 从已有日历的最后一年开始，以获取该年剩余的交易日
        start_year = pd.Timestamp(daily_calendar.iloc[-1, 0]).year

    # 从Baostock查询从开始年份到当前年份年底的所有交易日
    rs = bs.query_trade_dates(start_date=f"{start_year}-01-01", end_date=f"{end_year}-12-31")
    data_list = []
    while (rs.error_code == "0") & rs.next():
        _row_data = rs.get_row_data()
        if int(_row_data[1]) == 1:  # is_trading_day == 1
            data_list.append(_row_data[0])

    bs.logout() # 登出Baostock

    if not data_list:
        logger.warning("未能从Baostock获取到任何交易日数据。")
        return

    # 根据频率生成最终的日历列表
    future_date_list = generate_qlib_calendar(sorted(data_list), freq=freq)

    # 将新获取的日历与旧日历合并、去重、排序
    if not daily_calendar.empty:
        existing_dates = generate_qlib_calendar(daily_calendar.iloc[:, 0].tolist(), freq=freq)
        combined_list = sorted(set(existing_dates + future_date_list))
    else:
        combined_list = sorted(set(future_date_list))

    # 写入文件
    write_calendar_to_qlib(qlib_dir, combined_list, freq=freq)
    logger.info(f"成功获取并合并了从 {start_year}-01-01 到 {end_year}-12-31 的交易日历。")


if __name__ == "__main__":
    fire.Fire(future_calendar_collector)
