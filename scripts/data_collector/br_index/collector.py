# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

# 导入必要的库
from functools import partial
import sys
from pathlib import Path
import datetime

import fire
import pandas as pd
from tqdm import tqdm
from loguru import logger

# 将上两级目录添加到系统路径，以便导入所需的模块
CUR_DIR = Path(__file__).resolve().parent
sys.path.append(str(CUR_DIR.parent.parent))

from data_collector.index import IndexBase
from data_collector.utils import get_instruments

# 巴西股市IBOVESPA指数的成分股调整周期为每四个月一次，这里定义了每个周期的起始月份
# 1Q: 1-4月 (起始日01-03), 2Q: 5-8月 (起始日05-01), 3Q: 9-12月 (起始日09-01)
quarter_dict = {"1Q": "01-03", "2Q": "05-01", "3Q": "09-01"}


class IBOVIndex(IndexBase):
    """
    用于收集巴西IBOVESPA指数成分股信息的类。
    数据源自一个GitHub仓库，该仓库记录了指数的历史成分。
    """
    # GitHub仓库中历史成分股文件的URL模板
    ibov_index_composition = "https://raw.githubusercontent.com/igor17400/IBOV-HCI/main/historic_composition/{}.csv"
    years_4_month_periods = []

    def __init__(
        self,
        index_name: str,
        qlib_dir: [str, Path] = None,
        freq: str = "day",
        request_retry: int = 5,
        retry_sleep: int = 3,
    ):
        super(IBOVIndex, self).__init__(
            index_name=index_name, qlib_dir=qlib_dir, freq=freq, request_retry=request_retry, retry_sleep=retry_sleep
        )

        self.today: datetime.date = datetime.date.today()
        # 根据当前月份计算所属的四个月周期
        self.current_4_month_period = self.get_current_4_month_period(self.today.month)
        self.year = str(self.today.year)
        # 生成从2003年至今的所有四个月周期列表
        self.years_4_month_periods = self.get_four_month_period()

    @property
    def bench_start_date(self) -> pd.Timestamp:
        """
        IBOVESPA指数始于1968年1月2日，但目前找到的数据源仅从2003年第一季度开始记录。
        因此，我们将基准开始日期设为2003年1月3日。
        """
        return pd.Timestamp("2003-01-03")

    def get_current_4_month_period(self, current_month: int) -> str:
        """
        根据当前月份计算其所属的四个月周期。
        巴西股市使用Q(Quarter)代表四个月的周期，不同于英语中代表三个月的季度。
        1-4月: 1Q, 5-8月: 2Q, 9-12月: 3Q
        """
        if current_month < 5:
            return "1Q"
        if current_month < 9:
            return "2Q"
        if current_month <= 12:
            return "3Q"
        return "-1" # 表示错误

    def get_four_month_period(self) -> list:
        """
        生成从2003年至今的所有历史四个月周期列表。
        格式为 '年份_周期'，例如 '2003_1Q'。
        """
        four_months_period = ["1Q", "2Q", "3Q"]
        init_year = 2003
        now = datetime.datetime.now()
        current_year = now.year
        current_month = now.month

        # 遍历历史年份
        for year in range(init_year, current_year):
            for el in four_months_period:
                self.years_4_month_periods.append(f"{year}_{el}")

        # 处理当前年份
        current_4_month_period = self.get_current_4_month_period(current_month)
        for i in range(int(current_4_month_period[0])):
            self.years_4_month_periods.append(f"{current_year}_{i+1}Q")

        return self.years_4_month_periods

    def format_datetime(self, inst_df: pd.DataFrame) -> pd.DataFrame:
        """格式化instrument文件中的日期时间列。"""
        logger.info("正在格式化日期时间...")
        if self.freq != "day":
            # 对于非日线频率，结束时间设置为当天的23:59
            inst_df[self.END_DATE_FIELD] = inst_df[self.END_DATE_FIELD].apply(
                lambda x: (pd.Timestamp(x) + pd.Timedelta(hours=23, minutes=59)).strftime("%Y-%m-%d %H:%M:%S")
            )
        else:
            # 对于日线频率，格式化为 YYYY-MM-DD
            inst_df[self.START_DATE_FIELD] = inst_df[self.START_DATE_FIELD].apply(
                lambda x: pd.Timestamp(x).strftime("%Y-%m-%d")
            )
            inst_df[self.END_DATE_FIELD] = inst_df[self.END_DATE_FIELD].apply(
                lambda x: pd.Timestamp(x).strftime("%Y-%m-%d")
            )
        return inst_df

    def format_quarter(self, cell: str) -> str:
        """
        将 '年份_周期' 格式的字符串转换为 'YYYY-MM-DD' 格式的日期字符串。
        """
        cell_split = cell.split("_")
        return f"{cell_split[0]}-{quarter_dict[cell_split[1]]}"

    def get_changes(self) -> pd.DataFrame:
        """
        通过比较相邻两个周期的成分股列表，获取成分股的调入和调出记录。
        """
        logger.info(f"正在获取 {self.index_name} 指数的成分股变动...")

        try:
            df_changes_list = []
            # 遍历所有历史周期（除了最后一个）
            for i in tqdm(range(len(self.years_4_month_periods) - 1)):
                # 读取当前周期和下一个周期的成分股列表
                df_current = pd.read_csv(
                    self.ibov_index_composition.format(self.years_4_month_periods[i]), on_bad_lines="skip"
                )["symbol"]
                df_next = pd.read_csv(
                    self.ibov_index_composition.format(self.years_4_month_periods[i + 1]), on_bad_lines="skip"
                )["symbol"]

                # 找出被移除的股票（存在于当前周期，但不存在于下一周期）
                remove_date = self.format_quarter(self.years_4_month_periods[i])
                list_remove = list(df_current[~df_current.isin(df_next)])
                df_removed = pd.DataFrame(
                    {"date": remove_date, "type": "remove", "symbol": list_remove}
                )

                # 找出新增的股票（存在于下一周期，但不存在于当前周期）
                add_date = self.format_quarter(self.years_4_month_periods[i + 1])
                list_add = list(df_next[~df_next.isin(df_current)])
                df_added = pd.DataFrame(
                    {"date": add_date, "type": "add", "symbol": list_add}
                )

                # 合并调入和调出记录
                df_changes_list.append(pd.concat([df_added, df_removed], sort=False))

            df = pd.concat(df_changes_list).reset_index(drop=True)
            # 为股票代码添加 ".SA" 后缀，以符合Yahoo Finance的格式
            df["symbol"] = df["symbol"].astype(str) + ".SA"

            return df

        except Exception as E:
            logger.error(f"下载2008年指数成分时出错 - {E}")
            return pd.DataFrame()


    def get_new_companies(self) -> pd.DataFrame:
        """
        获取最新的指数成分股列表。
        """
        logger.info(f"正在获取 {self.index_name} 指数的最新成分股...")

        try:
            # 读取当前周期的成分股和首次纳入日期文件
            df_index = pd.read_csv(
                self.ibov_index_composition.format(f"{self.year}_{self.current_4_month_period}"), on_bad_lines="skip"
            )
            df_date_first_added = pd.read_csv(
                self.ibov_index_composition.format(f"date_first_added_{self.year}_{self.current_4_month_period}"),
                on_bad_lines="skip",
            )

            # 合并两个文件
            df = df_index.merge(df_date_first_added, on="symbol")[["symbol", "Date First Added"]]
            # 设置开始日期为首次纳入日期
            df[self.START_DATE_FIELD] = df["Date First Added"].map(self.format_quarter)
            # 设置结束日期为当前周期的开始日期
            df[self.END_DATE_FIELD] = f"{self.year}-{quarter_dict[self.current_4_month_period]}"

            df = df[["symbol", self.START_DATE_FIELD, self.END_DATE_FIELD]]
            # 添加.SA后缀
            df["symbol"] = df["symbol"].astype(str) + ".SA"

            return df

        except Exception as E:
            logger.error(f"获取最新成分股时出错 - {E}")
            return pd.DataFrame()

    def filter_df(self, df: pd.DataFrame) -> pd.DataFrame:
        """一个辅助函数，用于从DataFrame中筛选出'Código'列。"""
        if "Código" in df.columns:
            return df.loc[:, ["Código"]].copy()


if __name__ == "__main__":
    # 使用fire库将get_instruments函数（已通过partial包装）暴露为命令行工具
    fire.Fire(partial(get_instruments, market_index="br_index"))
