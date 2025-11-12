# 获取未来交易日

## 简介

本脚本用于获取未来的交易日历，并将其保存为一个单独的文件（例如 `day_future.txt`）。

在 `Qlib` 中，你可以通过 `D.calendar(future=True)` 来加载这个包含了未来交易日的日历，这对于需要预测未来日期或进行相关研究的场景非常有用。

脚本会执行以下操作：
1.  读取 `Qlib` 数据目录中已有的日历（例如 `day.txt`）。
2.  从 `baostock` 获取从现有日历的最后一年到当前年份年底的所有交易日。
3.  将旧日历和新获取的日历合并、去重。
4.  将合并后的完整日历保存到 `calendars` 目录下的 `day_future.txt` 或 `1min_future.txt` 文件中。

## 环境要求

在运行脚本之前，请先安装所需的依赖包：
```bash
pip install -r requirements.txt
```

## 数据收集命令

```bash
# 获取日线级别的未来交易日历
python future_trading_date_collector.py future_calendar_collector --qlib_dir ~/.qlib/qlib_data/cn_data --freq day

# 获取分钟线级别的未来交易日历
python future_trading_date_collector.py future_calendar_collector --qlib_dir ~/.qlib/qlib_data/cn_data --freq 1min
```

## 参数说明

-   `qlib_dir`: **（必需）** 你的Qlib数据目录路径。脚本会读取此目录下的现有日历，并将结果保存到此目录中。
-   `freq`: 日历的频率, 可选值为 [`day`, `1min`], 默认为 `day`。
