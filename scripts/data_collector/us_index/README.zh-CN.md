# 纳斯达克100/标普500/标普400/道琼斯工业平均指数历史成分股收集

## 依赖要求

```bash
pip install -r requirements.txt
```

## 收集器数据

```bash
# 解析成分股，在 qlib/instruments 中使用。
python collector.py --index_name SP500 --qlib_dir ~/.qlib/qlib_data/us_data --method parse_instruments

# 解析新公司
python collector.py --index_name SP500 --qlib_dir ~/.qlib/qlib_data/us_data --method save_new_companies

# index_name 支持: SP500, NASDAQ100, DJIA, SP400
# 帮助
python collector.py --help
```
