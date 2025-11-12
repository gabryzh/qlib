# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

# 导入必要的库
import re
import abc
import sys
from io import BytesIO
from typing import List, Iterable
from pathlib import Path

import fire
import requests
import pandas as pd
import baostock as bs  # 用于获取中证500历史成分股
from tqdm import tqdm
from loguru import logger

# 将上两级目录添加到系统路径
CUR_DIR = Path(__file__).resolve().parent
sys.path.append(str(CUR_DIR.parent.parent))

from data_collector.index import IndexBase
from data_collector.utils import get_calendar_list, get_trading_date_by_shift, deco_retry
from data_collector.utils import get_instruments

# 中证指数官网最新的指数成分股下载链接模板
NEW_COMPANIES_URL = (
    "https://oss-ch.csindex.com.cn/static/html/csindex/public/uploads/file/autofile/cons/{index_code}cons.xls"
)

# 中证指数官网搜索指数调整公告的API链接模板
INDEX_CHANGES_URL = "https://www.csindex.com.cn/csindex-home/search/search-content?lang=cn&searchInput=%E5%85%B3%E4%BA%8E%E8%B0%83%E6%95%B4%E6%B2%AA%E6%B7%B1300%E5%92%8C%E4%B8%AD%E8%AF%81%E9%A6%99%E6%B8%AF100%E7%AD%89%E6%8C%87%E6%95%B0%E6%A0%B7%E6%9C%AC&pageNum={page_num}&pageSize={page_size}&sortField=date&dateRange=all&contentType=announcement"

# 请求头，模拟浏览器访问
REQ_HEADERS = {
    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.101 Safari/537.36 Edg/91.0.864.48"
}


@deco_retry
def retry_request(url: str, method: str = "get", exclude_status: List = None):
    """带重试功能的请求函数"""
    if exclude_status is None:
        exclude_status = []
    method_func = getattr(requests, method)
    _resp = method_func(url, headers=REQ_HEADERS, timeout=None)
    _status = _resp.status_code
    # 如果状态码不是200且不在排除列表内，则抛出异常以触发重试
    if _status not in exclude_status and _status != 200:
        raise ValueError(f"response status: {_status}, url={url}")
    return _resp


class CSIIndex(IndexBase):
    """
    获取中证(CSI)系列指数成分股的基类。
    """
    @property
    def calendar_list(self) -> List[pd.Timestamp]:
        """获取交易日历（带缓存）。"""
        _calendar = getattr(self, "_calendar_list", None)
        if not _calendar:
            _calendar = get_calendar_list(bench_code=self.index_name.upper())
            setattr(self, "_calendar_list", _calendar)
        return _calendar

    @property
    def new_companies_url(self) -> str:
        """获取最新成分股的URL。"""
        return NEW_COMPANIES_URL.format(index_code=self.index_code)

    @property
    def changes_url(self) -> str:
        """获取成分股变更公告的URL。"""
        return INDEX_CHANGES_URL

    @property
    @abc.abstractmethod
    def bench_start_date(self) -> pd.Timestamp:
        """抽象属性：指数的起始日期。"""
        raise NotImplementedError("必须重写 bench_start_date")

    @property
    @abc.abstractmethod
    def index_code(self) -> str:
        """抽象属性：指数的代码。"""
        raise NotImplementedError("必须重写 index_code")

    @property
    @abc.abstractmethod
    def html_table_index(self) -> int:
        """
        抽象属性：在变更公告HTML页面中，目标指数的表格是第几个。
        例如，沪深300通常是第一个(0)，中证100是第二个(1)。
        """
        raise NotImplementedError("必须重写 html_table_index")

    def format_datetime(self, inst_df: pd.DataFrame) -> pd.DataFrame:
        """格式化instrument文件中的日期时间列。"""
        if self.freq != "day":
            inst_df[self.START_DATE_FIELD] = inst_df[self.START_DATE_FIELD].apply(
                lambda x: (pd.Timestamp(x) + pd.Timedelta(hours=9, minutes=30)).strftime("%Y-%m-%d %H:%M:%S")
            )
            inst_df[self.END_DATE_FIELD] = inst_df[self.END_DATE_FIELD].apply(
                lambda x: (pd.Timestamp(x) + pd.Timedelta(hours=15, minutes=0)).strftime("%Y-%m-%d %H:%M:%S")
            )
        return inst_df

    def get_changes(self) -> pd.DataFrame:
        """获取成分股历史变动。"""
        logger.info("开始获取成分股历史变动...")
        res = []
        # 遍历所有找到的变更公告URL
        for _url in self._get_change_notices_url():
            _df = self._read_change_from_url(_url)
            if not _df.empty:
                res.append(_df)
        logger.info("获取成分股历史变动结束。")
        return pd.concat(res, sort=False) if res else pd.DataFrame()


    @staticmethod
    def normalize_symbol(symbol: str) -> str:
        """规范化股票代码，添加'SH'或'SZ'前缀。"""
        symbol = f"{int(symbol):06d}"
        return f"SH{symbol}" if symbol.startswith("60") or symbol.startswith("688") else f"SZ{symbol}"

    def _parse_excel(self, excel_url: str, add_date: pd.Timestamp, remove_date: pd.Timestamp) -> pd.DataFrame:
        """从Excel文件中解析成分股变动。"""
        content = retry_request(excel_url, exclude_status=[404]).content
        _io = BytesIO(content)
        # 将下载的Excel文件缓存到本地
        with self.cache_dir.joinpath(
            f"{self.index_name.lower()}_changes_{add_date.strftime('%Y%m%d')}.{excel_url.split('.')[-1]}"
        ).open("wb") as fp:
            fp.write(content)

        # 读取Excel中所有sheet
        df_map = pd.read_excel(_io, sheet_name=None, engine='openpyxl') # 指定引擎

        tmp = []
        # 分别处理“调入”和“调出”两个sheet
        for _s_name, _type, _date in [("调入", self.ADD, add_date), ("调出", self.REMOVE, remove_date)]:
            if _s_name in df_map:
                _df = df_map[_s_name]
                # 筛选出对应指数代码的记录
                _df = _df.loc[_df["指数代码"] == self.index_code, ["证券代码"]]
                _df = _df.applymap(self.normalize_symbol)
                _df.columns = [self.SYMBOL_FIELD_NAME]
                _df["type"] = _type
                _df[self.DATE_FIELD_NAME] = _date
                tmp.append(_df)
        return pd.concat(tmp) if tmp else pd.DataFrame()


    def _parse_table(self, content: str, add_date: pd.Timestamp, remove_date: pd.Timestamp) -> pd.DataFrame:
        """从HTML内容中解析成分股变动的表格。"""
        df = pd.DataFrame()
        _tmp_count = 0
        try:
            # 使用pandas.read_html解析所有表格
            tables = pd.read_html(content)
            for _df in tables:
                # 简单的校验，判断是否为目标表格
                if _df.shape[-1] != 4 or _df.isnull().iloc[0, 0]:
                    continue
                _tmp_count += 1
                if self.html_table_index + 1 > _tmp_count:
                    continue

                tmp = []
                # 分别解析调出和调入列表
                for _series, _type, _date in [
                    (_df.iloc[2:, 0], self.REMOVE, remove_date), # 第0列为调出
                    (_df.iloc[2:, 2], self.ADD, add_date),     # 第2列为调入
                ]:
                    _tmp_df = pd.DataFrame()
                    _tmp_df[self.SYMBOL_FIELD_NAME] = _series.map(self.normalize_symbol)
                    _tmp_df["type"] = _type
                    _tmp_df[self.DATE_FIELD_NAME] = _date
                    tmp.append(_tmp_df.dropna())
                df = pd.concat(tmp)
                # 缓存解析结果
                df.to_csv(
                    str(self.cache_dir.joinpath(f"{self.index_name.lower()}_changes_{add_date.strftime('%Y%m%d')}.csv").resolve())
                )
                break # 找到目标表格后即退出
        except Exception as e:
            logger.warning(f"解析HTML表格失败: {e}")
        return df

    def _read_change_from_url(self, url: str) -> pd.DataFrame:
        """从单个公告URL中读取成分股变动信息。"""
        resp = retry_request(url).json()["data"]
        title = resp["title"]
        # 根据标题过滤不相关的公告
        if not title.startswith("关于") or "沪深300" not in title: # 很多公告标题都包含沪深300
            return pd.DataFrame()

        logger.info(f"正在加载公告: {title}")
        _text = resp["content"]

        # 从公告正文中解析生效日期
        date_list = re.findall(r"(\d{4}).*?年.*?(\d+).*?月.*?(\d+).*?日", _text)
        if len(date_list) >= 2:
            add_date = pd.Timestamp("-".join(date_list[0]))
        else:
            # 兼容只有年月的格式
            _date_str = re.findall(r"(\d{4}).*?年.*?(\d+).*?月", _text)[0]
            _date = pd.Timestamp(f"{_date_str[0]}-{_date_str[1]}")
            add_date = get_trading_date_by_shift(self.calendar_list, _date, shift=0)

        # 如果公告中提到“盘后”或“市后”生效，则生效日为下一个交易日
        if "盘后" in _text or "市后" in _text:
            add_date = get_trading_date_by_shift(self.calendar_list, add_date, shift=1)
        remove_date = get_trading_date_by_shift(self.calendar_list, add_date, shift=-1)

        # 尝试寻找公告中的Excel附件链接
        excel_url = None
        if resp.get("enclosureList"):
            excel_url = resp["enclosureList"][0]["fileUrl"]
        else:
            excel_url_list = re.findall('.*href="(.*?xls.*?)".*', _text)
            if excel_url_list:
                excel_url = excel_url_list[0]
                if not excel_url.startswith("http"):
                    excel_url = f"http://www.csindex.com.cn{excel_url if excel_url.startswith('/') else '/' + excel_url}"

        # 优先从Excel解析，失败或不存在则从HTML表格解析
        if excel_url:
            try:
                logger.info(f"尝试从Excel解析 {add_date.date()} 的变动...")
                return self._parse_excel(excel_url, add_date, remove_date)
            except Exception:
                logger.warning("从Excel解析失败，尝试从网页表格解析...")
                return self._parse_table(_text, add_date, remove_date)
        else:
            logger.info(f"未找到Excel附件，尝试从网页表格解析 {add_date.date()} 的变动...")
            return self._parse_table(_text, add_date, remove_date)

    def _get_change_notices_url(self) -> Iterable[str]:
        """获取所有历史变更公告的URL列表。"""
        # 先请求一次获取总条数，再请求所有数据
        page_num = 1
        page_size = 5
        data = retry_request(self.changes_url.format(page_size=page_size, page_num=page_num)).json()
        total = data.get("total", 0)
        if total == 0:
            return

        data = retry_request(self.changes_url.format(page_size=total, page_num=page_num)).json()
        for item in data.get("data", []):
            yield f"https://www.csindex.com.cn/csindex-home/announcement/queryAnnouncementById?id={item['id']}"

    def get_new_companies(self) -> pd.DataFrame:
        """获取最新的指数成分股列表。"""
        logger.info("正在获取最新成分股...")
        context = retry_request(self.new_companies_url).content
        # 缓存文件
        with self.cache_dir.joinpath(
            f"{self.index_name.lower()}_new_companies.{self.new_companies_url.split('.')[-1]}"
        ).open("wb") as fp:
            fp.write(context)

        _io = BytesIO(context)
        df = pd.read_excel(_io, engine='openpyxl')
        # 提取所需列
        df = df.iloc[:, [0, 4]]
        df.columns = [self.END_DATE_FIELD, self.SYMBOL_FIELD_NAME]
        df[self.SYMBOL_FIELD_NAME] = df[self.SYMBOL_FIELD_NAME].map(self.normalize_symbol)
        df[self.END_DATE_FIELD] = pd.to_datetime(df[self.END_DATE_FIELD].astype(str))
        df[self.START_DATE_FIELD] = self.bench_start_date
        logger.info("获取最新成分股结束。")
        return df

# --- 具体指数的实现类 ---

class CSI300Index(CSIIndex):
    """沪深300指数"""
    @property
    def index_code(self): return "000300"
    @property
    def bench_start_date(self): return pd.Timestamp("2005-01-01")
    @property
    def html_table_index(self): return 0

class CSI100Index(CSIIndex):
    """中证100指数"""
    @property
    def index_code(self): return "000903"
    @property
    def bench_start_date(self): return pd.Timestamp("2006-05-29")
    @property
    def html_table_index(self): return 1

class CSI500Index(CSIIndex):
    """中证500指数"""
    @property
    def index_code(self): return "000905"
    @property
    def bench_start_date(self): return pd.Timestamp("2007-01-15")

    def get_changes(self) -> pd.DataFrame:
        """
        中证500的变更公告不规范，无法用CSI300的方法解析。
        此处改用baostock获取每日历史成分股，然后通过比对得出变更。
        """
        return self.get_changes_with_history_companies(self.get_history_companies())

    def get_history_companies(self) -> pd.DataFrame:
        """使用baostock按周获取历史成分股。"""
        bs.login()
        today = pd.Timestamp.now()
        # 创建一个每周的日期范围进行查询
        date_range = pd.date_range(start="2007-01-15", end=today, freq="7D")
        ret_list = []
        for date in tqdm(date_range, desc="正在从Baostock下载中证500历史成分股"):
            result = self.get_data_from_baostock(date.strftime("%Y-%m-%d"))
            ret_list.append(result[["date", "symbol"]])
        bs.logout()
        return pd.concat(ret_list, sort=False).drop_duplicates()


    @staticmethod
    def get_data_from_baostock(date) -> pd.DataFrame:
        """从baostock获取指定日期的中证500成分股。"""
        col = ["date", "symbol", "code_name"]
        rs = bs.query_zz500_stocks(date=str(date))
        zz500_stocks = []
        while (rs.error_code == "0") & rs.next():
            zz500_stocks.append(rs.get_row_data())
        result = pd.DataFrame(zz500_stocks, columns=col)
        result["symbol"] = result["symbol"].apply(lambda x: x.replace(".", "").upper())
        return result

    def get_new_companies(self) -> pd.DataFrame:
        """从中证官网获取最新成分股。"""
        # 注意：这里的实现与基类相同，但保留以备将来可能的特化。
        return super().get_new_companies()


if __name__ == "__main__":
    fire.Fire(get_instruments)
