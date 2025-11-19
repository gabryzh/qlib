# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

import unittest
from unittest.mock import MagicMock, patch
import pandas as pd
import numpy as np
from pathlib import Path
import sys
import os
import shutil

CUR_DIR = Path(__file__).resolve().parent
sys.path.append(str(CUR_DIR.parent))
from scripts.data_collector.tushare.collector import TushareCollector, TushareNormalize1d, TushareNormalize1min, Run


class TestTushareCollector(unittest.TestCase):
    def setUp(self):
        self.mock_pro = MagicMock()
        self.start_date = "2022-01-01"
        self.end_date = "2022-01-05"
        self.ts_code = "000001.SZ"
        self.save_dir = Path("/tmp/test_tushare")
        self.save_dir.mkdir(exist_ok=True)

        # Sample data for pro_bar (daily)
        self.pro_bar_daily_data = pd.DataFrame({
            "ts_code": [self.ts_code] * 2,
            "trade_date": ["20220104", "20220105"],
            "open": [10, 11], "high": [10.5, 11.5], "low": [9.5, 10.5], "close": [10.2, 11.2],
            "pre_close": [10, 10.2], "change": [0.2, 1.0], "pct_chg": [2, 9.8],
            "vol": [1000, 1100], "amount": [10000, 12320],
        })

        # Sample data for adj_factor
        self.adj_factor_data = pd.DataFrame({
            "ts_code": [self.ts_code] * 2,
            "trade_date": ["20220104", "20220105"],
            "adj_factor": [1.0, 1.0],
        })

        # Sample data for pro_bar (minute)
        self.pro_bar_minute_data = pd.DataFrame({
            "ts_code": [self.ts_code] * 2,
            "trade_time": ["2022-01-04 09:31:00", "2022-01-04 09:32:00"],
            "open": [10, 10.1], "high": [10.2, 10.2], "low": [9.9, 10.0], "close": [10.1, 10.1],
            "vol": [100, 110], "amount": [1000, 1111],
        })
        
        # Sample data for index
        self.index_data = pd.DataFrame({
            "ts_code": ["000300.SH"] * 2,
            "trade_date": ["20220104", "20220105"],
            "open": [5000, 5010], "high": [5050, 5050], "low": [4950, 4980], "close": [5020, 5030],
            "pre_close": [5000, 5020], "change": [20, 10], "pct_chg": [0.4, 0.2],
            "vol": [100000, 110000], "amount": [1000000, 1100000],
        })

    @patch("scripts.data_collector.tushare.collector.get_hs_stock_symbols")
    @patch("tushare.pro_api")
    def test_collect_day_data(self, mock_pro_api, mock_get_symbols):
        mock_get_symbols.return_value = [self.ts_code]
        self.mock_pro.pro_bar.return_value = self.pro_bar_daily_data
        self.mock_pro.adj_factor.return_value = self.adj_factor_data
        mock_pro_api.return_value = self.mock_pro

        collector = TushareCollector(
            save_dir=self.save_dir, start=self.start_date, end=self.end_date, tushare_token="test"
        )
        
        df = collector.get_data(self.ts_code, "1d", self.start_date, self.end_date)
        
        self.assertFalse(df.empty)
        self.assertEqual(len(df), 2)
        self.assertIn("adj_factor", df.columns)

        normalizer = TushareNormalize1d()
        normalized_df = normalizer.normalize(df)

        self.assertIn("factor", normalized_df.columns)
        self.assertNotIn("adj_factor", normalized_df.columns)

    @patch("scripts.data_collector.tushare.collector.get_hs_stock_symbols")
    @patch("tushare.pro_api")
    def test_collect_minute_data(self, mock_pro_api, mock_get_symbols):
        mock_get_symbols.return_value = [self.ts_code]
        self.mock_pro.pro_bar.return_value = self.pro_bar_minute_data
        mock_pro_api.return_value = self.mock_pro

        collector = TushareCollector(
            save_dir=self.save_dir, start="2022-01-04 09:30:00", end="2022-01-04 09:33:00", 
            interval="1min", tushare_token="test"
        )
        
        df = collector.get_data(self.ts_code, "1min", "2022-01-04 09:30:00", "2022-01-04 09:33:00")
        
        self.assertFalse(df.empty)
        self.assertEqual(len(df), 2)
        self.assertIn("trade_time", df.columns)

        normalizer = TushareNormalize1min()
        with patch('scripts.data_collector.tushare.collector.get_calendar_list') as mock_get_calendar:
            mock_get_calendar.return_value = pd.to_datetime(["2022-01-04"]).to_series()
            normalized_df = normalizer.normalize(df)
            self.assertIn("date", normalized_df.columns)

    @patch("scripts.data_collector.tushare.collector.get_hs_stock_symbols")
    @patch("tushare.pro_api")
    def test_download_index_data(self, mock_pro_api, mock_get_symbols):
        mock_get_symbols.return_value = [self.ts_code]
        self.mock_pro.pro_bar.return_value = self.index_data
        mock_pro_api.return_value = self.mock_pro

        collector = TushareCollector(
            save_dir=self.save_dir, start=self.start_date, end=self.end_date, tushare_token="test"
        )
        collector.download_index_data()

        index_file = self.save_dir / "sh000300.csv"
        self.assertTrue(index_file.exists())
        df = pd.read_csv(index_file)
        self.assertEqual(len(df), 2)
        self.assertEqual(df["symbol"].iloc[0], "sh000300")

    def tearDown(self):
        shutil.rmtree(self.save_dir)

if __name__ == "__main__":
    unittest.main()
