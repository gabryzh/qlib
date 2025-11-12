# 导入sys模块，用于操作Python运行时环境
import sys
# 导入abc模块，用于定义抽象基类
import abc
# 从pathlib导入Path类，用于处理文件路径
from pathlib import Path
# 从typing导入List，用于类型注解
from typing import List

# 导入pandas库，用于数据处理
import pandas as pd
# 导入tqdm库，用于显示进度条
from tqdm import tqdm
# 导入loguru库，用于日志记录
from loguru import logger

# 将当前文件的父目录添加到系统路径中，以便导入同级模块
CUR_DIR = Path(__file__).resolve().parent
sys.path.append(str(CUR_DIR.parent))


# 从data_collector.utils模块导入工具函数
from data_collector.utils import get_trading_date_by_shift


class IndexBase:
    """
    获取指数成分股及其历史变动的基类。
    所有具体的指数收集器（如CSI300, CSI500等）都应继承此类。
    """
    # 默认的结束日期，表示至今仍是成分股
    DEFAULT_END_DATE = pd.Timestamp("2099-12-31")
    # 定义DataFrame中常用的列名常量
    SYMBOL_FIELD_NAME = "symbol"
    DATE_FIELD_NAME = "date"
    START_DATE_FIELD = "start_date"
    END_DATE_FIELD = "end_date"
    CHANGE_TYPE_FIELD = "type"
    INSTRUMENTS_COLUMNS = [SYMBOL_FIELD_NAME, START_DATE_FIELD, END_DATE_FIELD]
    # 定义成分股变动类型
    REMOVE = "remove" # 移除
    ADD = "add"       # 新增
    # 股票代码前缀（例如'sh', 'sz'），子类可覆盖
    INST_PREFIX = ""

    def __init__(
        self,
        index_name: str,
        qlib_dir: [str, Path] = None,
        freq: str = "day",
        request_retry: int = 5,
        retry_sleep: int = 3,
    ):
        """
        初始化。

        Parameters
        ----------
        index_name: str
            指数名称 (例如 "CSI300")。
        qlib_dir: str or Path
            qlib数据目录。
        freq: str
            数据频率, "day" 或 "1min"。
        request_retry: int
            请求失败时的重试次数。
        retry_sleep: int
            每次重试前的等待时间（秒）。
        """
        self.index_name = index_name
        if qlib_dir is None:
            # 如果未提供qlib目录，则使用默认路径
            qlib_dir = Path(__file__).resolve().parent.joinpath("qlib_data")
        # instruments文件的存放目录
        self.instruments_dir = Path(qlib_dir).expanduser().resolve().joinpath("instruments")
        self.instruments_dir.mkdir(exist_ok=True, parents=True)
        # 用于缓存下载的原始网页或数据的目录
        self.cache_dir = Path(f"~/.cache/qlib/index/{self.index_name}").expanduser().resolve()
        self.cache_dir.mkdir(exist_ok=True, parents=True)
        self._request_retry = request_retry
        self._retry_sleep = retry_sleep
        self.freq = freq

    @property
    @abc.abstractmethod
    def bench_start_date(self) -> pd.Timestamp:
        """
        抽象属性：指数的起始日期。

        Returns
        -------
            pd.Timestamp: 指数的起始日期。
        """
        raise NotImplementedError("必须重写 bench_start_date")

    @property
    @abc.abstractmethod
    def calendar_list(self) -> List[pd.Timestamp]:
        """
        抽象属性：获取历史交易日历。

        Returns
        -------
            List[pd.Timestamp]: 交易日历列表。
        """
        raise NotImplementedError("必须重写 calendar_list")

    @abc.abstractmethod
    def get_new_companies(self) -> pd.DataFrame:
        """
        抽象方法：获取最新的指数成分股列表。

        Returns
        -------
            pd.DataFrame:
                DataFrame结构应为:
                    symbol     start_date    end_date
                    SH600000   2000-01-01    2099-12-31
                数据类型:
                    symbol: str
                    start_date: pd.Timestamp
                    end_date: pd.Timestamp
        """
        raise NotImplementedError("必须重写 get_new_companies")

    @abc.abstractmethod
    def get_changes(self) -> pd.DataFrame:
        """
        抽象方法：获取指数成分股的历史变动。

        Returns
        -------
            pd.DataFrame:
                DataFrame结构应为:
                    symbol      date        type
                    SH600000  2019-11-11    add
                    SH600000  2020-11-10    remove
                数据类型:
                    symbol: str
                    date: pd.Timestamp
                    type: str, 值应为 "add" 或 "remove"
        """
        raise NotImplementedError("必须重写 get_changes")

    @abc.abstractmethod
    def format_datetime(self, inst_df: pd.DataFrame) -> pd.DataFrame:
        """
        抽象方法：格式化instrument文件中的日期时间列。

        Parameters
        ----------
        inst_df: pd.DataFrame
            包含 [symbol, start_date, end_date] 列的DataFrame。

        Returns
        -------
        pd.DataFrame
            格式化后的DataFrame。
        """
        raise NotImplementedError("必须重写 format_datetime")

    def save_new_companies(self):
        """
        保存最新的成分股列表到文件。
        文件名格式为: `<index_name>_only_new.txt`

        Examples
        -------
            $ python collector.py index save_new_companies --index_name CSI300 --qlib_dir ~/.qlib/qlib_data/cn_data
        """
        df = self.get_new_companies()
        if df is None or df.empty:
            raise ValueError(f"获取最新成分股失败: {self.index_name}")
        df = df.drop_duplicates([self.SYMBOL_FIELD_NAME])
        # 保存为tab分隔的无表头文件
        df.loc[:, self.INSTRUMENTS_COLUMNS].to_csv(
            self.instruments_dir.joinpath(f"{self.index_name.lower()}_only_new.txt"), sep="\t", index=False, header=None
        )

    def get_changes_with_history_companies(self, history_companies: pd.DataFrame) -> pd.DataFrame:
        """
        根据每日的成分股列表历史，推断出成分股的变动（调入/调出）。
        注意：这种方法可能不完全准确，因为有些指数的调整日和生效日可能不同。

        Parameters
        ----------
        history_companies : pd.DataFrame
            包含每日成分股列表的DataFrame。
            结构:
                symbol        date
                SH600000   2020-11-11
            数据类型:
                symbol: str
                date: pd.Timestamp

        Return
        --------
            pd.DataFrame:
                与 `get_changes` 方法返回的DataFrame结构相同。
        """
        logger.info("开始从历史成分股列表推断变动...")
        last_code = []
        result_df_list = []
        _columns = [self.DATE_FIELD_NAME, self.SYMBOL_FIELD_NAME, self.CHANGE_TYPE_FIELD]
        # 从最近的日期开始，倒序遍历
        for _trading_date in tqdm(sorted(history_companies[self.DATE_FIELD_NAME].unique(), reverse=True)):
            _current_code = history_companies[history_companies[self.DATE_FIELD_NAME] == _trading_date][
                self.SYMBOL_FIELD_NAME
            ].tolist()
            if last_code:
                # 与上一日的列表比较，得出新增和移除的股票
                add_code = list(set(last_code) - set(_current_code))
                remove_code = list(set(_current_code) - set(last_code))

                # 新增的股票，生效日期是当前日期的下一个交易日
                for _code in add_code:
                    result_df_list.append(
                        pd.DataFrame(
                            [[get_trading_date_by_shift(self.calendar_list, _trading_date, 1), _code, self.ADD]],
                            columns=_columns,
                        )
                    )
                # 移除的股票，生效日期是当前日期
                for _code in remove_code:
                    result_df_list.append(
                        pd.DataFrame(
                            [[_trading_date, _code, self.REMOVE]],
                            columns=_columns,
                        )
                    )
            last_code = _current_code
        df = pd.concat(result_df_list)
        logger.info("从历史成分股列表推断变动结束。")
        return df

    def parse_instruments(self):
        """
        解析并生成最终的instrument文件 (例如 csi300.txt)。
        该文件记录了每只成分股在指数中的完整生命周期（开始日期和结束日期）。

        Examples
        -------
            $ python collector.py index parse_instruments --index_name CSI300 --qlib_dir ~/.qlib/qlib_data/cn_data
        """
        logger.info(f"开始解析 {self.index_name.lower()} 的成分股...")
        instruments_columns = [self.SYMBOL_FIELD_NAME, self.START_DATE_FIELD, self.END_DATE_FIELD]

        # 获取历史变动和最新成分股
        changers_df = self.get_changes()
        new_df = self.get_new_companies()

        if new_df is None or new_df.empty:
            raise ValueError(f"获取最新成分股失败: {self.index_name}")
        new_df = new_df.copy()

        logger.info("根据历史变动反向推算每只股票的完整生命周期...")
        # 从最近的变动开始，倒序处理
        for _row in tqdm(changers_df.sort_values(self.DATE_FIELD_NAME, ascending=False).itertuples(index=False)):
            if _row.type == self.ADD:
                # 如果是新增(ADD)事件，说明这只股票在 `_row.date` 之前不在指数中。
                # 我们需要找到这只股票在 `new_df` 中对应的记录，并将其 `start_date` 更新为 `_row.date`。
                # 这表示它的生命周期是从 `_row.date` 开始的。
                min_end_date = new_df.loc[new_df[self.SYMBOL_FIELD_NAME] == _row.symbol, self.END_DATE_FIELD].min()
                new_df.loc[
                    (new_df[self.END_DATE_FIELD] == min_end_date) & (new_df[self.SYMBOL_FIELD_NAME] == _row.symbol),
                    self.START_DATE_FIELD,
                ] = _row.date
            else: # REMOVE
                # 如果是移除(REMOVE)事件，说明这只股票在 `_row.date` 之后就不在指数中了。
                # 这代表了一段完整的历史生命周期，从指数开始日期到 `_row.date`。
                # 我们在 `new_df` 中为它新增一条记录。
                _tmp_df = pd.DataFrame([[_row.symbol, self.bench_start_date, _row.date]], columns=instruments_columns)
                new_df = pd.concat([new_df, _tmp_df], sort=False)

        inst_df = new_df.loc[:, instruments_columns]
        _inst_prefix = self.INST_PREFIX.strip()
        if _inst_prefix:
            inst_df["save_inst"] = inst_df[self.SYMBOL_FIELD_NAME].apply(lambda x: f"{_inst_prefix}{x}")

        # 格式化日期并保存到文件
        inst_df = self.format_datetime(inst_df)
        inst_df.to_csv(
            self.instruments_dir.joinpath(f"{self.index_name.lower()}.txt"), sep="\t", index=False, header=None
        )
        logger.info(f"解析 {self.index_name.lower()} 成分股完成。")
