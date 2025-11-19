# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

import abc
import sys
import datetime
from pathlib import Path
from typing import Iterable

import fire
import numpy as np
import pandas as pd
import tushare as ts
from dateutil.tz import tzlocal
from loguru import logger

CUR_DIR = Path(__file__).resolve().parent
sys.path.append(str(CUR_DIR.parent.parent))

from data_collector.base import BaseCollector, BaseNormalize, BaseRun
from data_collector.utils import (
    deco_retry,
    get_calendar_list,
    get_hs_stock_symbols,
    generate_minutes_calendar_from_daily,
)


class TushareCollector(BaseCollector):
    retry = 5  # Configuration attribute.

    def __init__(
            self,
            save_dir: [str, Path],
            start=None,
            end=None,
            interval="1d",
            max_workers=4,
            max_collector_count=2,
            delay=0,
            check_data_length: int = None,
            limit_nums: int = None,
            tushare_token: str = None,  # Keep tushare_token as a parameter for flexibility, but don't enforce it
    ):
        super(TushareCollector, self).__init__(
            save_dir=save_dir,
            start=start,
            end=end,
            interval=interval,
            max_workers=max_workers,
            max_collector_count=max_collector_count,
            delay=delay,
            check_data_length=check_data_length,
            limit_nums=limit_nums,
        )
        # Assuming tushare token is already set globally or will be set by the user
        # If tushare_token is provided, set it. Otherwise, rely on global setting.
        if tushare_token:
            ts.set_token(tushare_token)
        self.pro = ts.pro_api()
        self.init_datetime()

    def init_datetime(self):
        if self.interval == self.INTERVAL_1min:
            self.start_datetime = max(self.start_datetime, self.DEFAULT_START_DATETIME_1MIN)
        elif self.interval == self.INTERVAL_1d:
            pass
        else:
            raise ValueError(f"interval error: {self.interval}")

        self.start_datetime = self.convert_datetime(self.start_datetime, self._timezone)
        self.end_datetime = self.convert_datetime(self.end_datetime, self._timezone)

    @staticmethod
    def convert_datetime(dt: [pd.Timestamp, datetime.date, str], timezone):
        try:
            dt = pd.Timestamp(dt, tz=timezone).timestamp()
            dt = pd.Timestamp(dt, tz=tzlocal(), unit="s")
        except ValueError as e:
            pass
        return dt

    def get_instrument_list(self):
        logger.info("get HS stock symbols......")
        symbols = get_hs_stock_symbols()
        logger.info(f"get {len(symbols)} symbols.")
        return symbols

    def normalize_symbol(self, symbol):
        symbol_s = symbol.split(".")
        symbol = f"sh{symbol_s[0]}" if symbol_s[-1] == "ss" else f"sz{symbol_s[0]}"
        return symbol

    @property
    def _timezone(self):
        return "Asia/Shanghai"

    @staticmethod
    @deco_retry
    def get_data_from_remote(pro, symbol, interval, start, end, asset='E'):
        if interval == '1d':
            _start_date = pd.Timestamp(start).strftime("%Y%m%d")
            _end_date = pd.Timestamp(end).strftime("%Y%m%d")
            freq = 'D'
            df = ts.pro_bar(
                ts_code=symbol,
                start_date=_start_date,
                end_date=_end_date,
                asset=asset,
                freq=freq,
            )
            if asset == 'E':
                adj_factor = pro.adj_factor(ts_code=symbol, start_date=_start_date, end_date=_end_date)
                if df is not None and not df.empty and adj_factor is not None and not adj_factor.empty:
                    df = pd.merge(df, adj_factor, on="trade_date", how="left")
            return df
        elif interval == '1min':
            _start_datetime = pd.Timestamp(start)
            _end_datetime = pd.Timestamp(end)
            _res = []
            while _start_datetime < _end_datetime:
                _tmp_end = min(_start_datetime + pd.Timedelta(days=7), _end_datetime)

                _start_date = _start_datetime.strftime("%Y-%m-%d %H:%M:%S")
                _end_date = _tmp_end.strftime("%Y-%m-%d %H:%M:%S")

                df = ts.pro_bar(
                    ts_code=symbol,
                    start_date=_start_date,
                    end_date=_end_date,
                    asset=asset,
                    freq='1min',
                )
                _res.append(df)
                _start_datetime = _tmp_end
            if _res:
                return pd.concat(_res, sort=False).sort_values(["ts_code", "trade_time"])

        raise ValueError(f"Unsupported interval: {interval}")

    def get_data(
            self, symbol: str, interval: str, start_datetime: pd.Timestamp, end_datetime: pd.Timestamp, asset: str = 'E'
    ) -> pd.DataFrame:
        self.sleep()
        resp = self.get_data_from_remote(
            self.pro,
            symbol,
            interval=interval,
            start=start_datetime,
            end=end_datetime,
            asset=asset
        )
        if resp is None or resp.empty:
            logger.warning(f"get data error: {symbol}--{start_datetime}--{end_datetime}")
            return pd.DataFrame()
        return resp

    def get_instrument_list(self):
        logger.info("get HS stock symbols......")
        symbols = get_hs_stock_symbols()
        logger.info(f"get {len(symbols)} symbols.")
        return symbols

    def normalize_symbol(self, symbol):
        return symbol.replace("SH", "sh").replace("SZ", "sz")

    def download_index_data(self):
        if self.interval == '1d':
            index_list = {
                "000300.SH": "sh000300",
                "000905.SH": "sh000905",
                "000016.SH": "sh000016",
            }
            for index_code, norm_code in index_list.items():
                logger.info(f"get bench data: {index_code}({norm_code})......")
                df = self.get_data(index_code, self.interval, self.start_datetime, self.end_datetime, asset='I')
                if not df.empty:
                    df.rename(columns={"ts_code": "symbol"}, inplace=True)
                    df["symbol"] = norm_code
                    _path = self.save_dir.joinpath(f"{norm_code}.csv")
                    if _path.exists():
                        _old_df = pd.read_csv(_path)
                        df = pd.concat([_old_df, df], sort=False)
                    df.to_csv(_path, index=False)

    def collector_data(self):
        super(TushareCollector, self).collector_data()
        self.download_index_data()


class TushareNormalize(BaseNormalize):
    def _get_calendar_list(self) -> Iterable[pd.Timestamp]:
        pass

    COLUMNS = ["open", "close", "high", "low", "volume", "amount"]

    @staticmethod
    def normalize_tushare(
            df: pd.DataFrame,
            calendar_list: list = None,
            date_field_name: str = "trade_date",
            symbol_field_name: str = "ts_code",
    ):
        if df.empty:
            return df

        df = df.copy()

        if "trade_time" in df.columns:
            df.rename(columns={"vol": "volume", "trade_time": "date", "ts_code": "symbol"}, inplace=True)
        else:
            df.rename(columns={"vol": "volume", "trade_date": "date", "ts_code": "symbol", "adj_factor": "factor"},
                      inplace=True)

        df["date"] = pd.to_datetime(df["date"])
        df.set_index("date", inplace=True)
        df = df[~df.index.duplicated(keep="first")]

        if calendar_list is not None:
            df = df.reindex(
                pd.DataFrame(index=calendar_list)
                .loc[
                    pd.Timestamp(df.index.min()).date(): pd.Timestamp(df.index.max()).date()
                                                         + pd.Timedelta(hours=23, minutes=59)
                ]
                .index
            )
        df.sort_index(inplace=True)
        df.loc[(df["volume"] <= 0) | np.isnan(df["volume"]), list(set(df.columns) - {"symbol"})] = np.nan

        if "symbol" in df.columns:
            df["symbol"] = df["symbol"].str.lower()
        df.index.names = ["date"]
        return df.reset_index()

    def normalize(self, df: pd.DataFrame) -> pd.DataFrame:
        df = self.normalize_tushare(df, self._calendar_list, self._date_field_name, self._symbol_field_name)
        return df


class TushareNormalize1d(TushareNormalize):
    def _get_calendar_list(self) -> Iterable[pd.Timestamp]:
        return get_calendar_list("ALL")

    def normalize(self, df: pd.DataFrame) -> pd.DataFrame:
        df = super().normalize(df)
        if "factor" in df.columns:
            df["factor"].fillna(method="ffill", inplace=True)
            for col in ["open", "close", "high", "low"]:
                df[col] = df[col] * df["factor"]
            df["volume"] = df["volume"] / df["factor"]
        return df


class TushareNormalize1min(TushareNormalize):
    AM_RANGE = ("09:30:00", "11:29:00")
    PM_RANGE = ("13:00:00", "14:59:00")

    def _get_calendar_list(self) -> Iterable[pd.Timestamp]:
        return generate_minutes_calendar_from_daily(get_calendar_list("ALL"), freq="1min", am_range=self.AM_RANGE,
                                                    pm_range=self.PM_RANGE)


class Run(BaseRun):
    def __init__(self, source_dir=None, normalize_dir=None, max_workers=1, interval="1d", tushare_token=None):
        super().__init__(source_dir, normalize_dir, max_workers, interval)
        self.tushare_token = tushare_token

    @property
    def collector_class_name(self):
        return "TushareCollector"

    @property
    def normalize_class_name(self):
        return f"TushareNormalize{self.interval}"

    @property
    def default_base_dir(self) -> [Path, str]:
        return CUR_DIR

    def download_data(self, max_collector_count=2, delay=0.5, start=None, end=None, check_data_length=None,
                      limit_nums=None, **kwargs):
        if self.interval == "1d" and pd.Timestamp(end) > pd.Timestamp(datetime.datetime.now().strftime("%Y-%m-%d")):
            raise ValueError(f"end_date: {end} is greater than the current date.")

        super(Run, self).download_data(max_collector_count, delay, start, end, check_data_length, limit_nums)

    def normalize_data(self, date_field_name: str = "trade_date", symbol_field_name: str = "ts_code", **kwargs):
        super(Run, self).normalize_data(date_field_name, symbol_field_name)


if __name__ == "__main__":
    fire.Fire(Run)
