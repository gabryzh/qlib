# 收集Point-in-Time数据

> *请**注意**，数据是从[baostock](http://baostock.com)收集的，数据可能并不完美。如果用户有高质量的数据集，我们建议他们准备自己的数据。更多信息，用户可以参考[相关文档](https://qlib.readthedocs.io/en/latest/component/data.html#converting-csv-format-into-qlib-format)*

## 依赖要求

```bash
pip install -r requirements.txt
```

## 收集器数据


### 下载季度A股数据

```bash
cd qlib/scripts/data_collector/pit/
# 从 baostock.com 下载
python collector.py download_data --source_dir ~/.qlib/stock_data/source/pit --start 2000-01-01 --end 2020-01-01 --interval quarterly
```

下载所有股票的数据非常耗时。如果您只想在几只股票上进行快速测试，可以运行以下命令
```bash
python collector.py download_data --source_dir ~/.qlib/stock_data/source/pit --start 2000-01-01 --end 2020-01-01 --interval quarterly --symbol_regex "^(600519|000725).*"
```


### 规范化数据
```bash
python collector.py normalize_data --interval quarterly --source_dir ~/.qlib/stock_data/source/pit --normalize_dir ~/.qlib/stock_data/source/pit_normalized
```



### 将数据转储为PIT格式

```bash
cd qlib/scripts
python dump_pit.py dump --data_path ~/.qlib/stock_data/source/pit_normalized --qlib_dir ~/.qlib/qlib_data/cn_data --interval quarterly
```
