# 收集加密货币数据

> *请**注意**，数据是从 [Coingecko](https://www.coingecko.com/en/api) 收集的，数据可能并不完美。如果用户有高质量的数据集，我们建议他们准备自己的数据。更多信息，用户可以参考[相关文档](https://qlib.readthedocs.io/en/latest/component/data.html#converting-csv-format-into-qlib-format)*

## 依赖要求

```bash
pip install -r requirements.txt
```

## 数据集的使用
> *由于缺少 OHLC 数据，加密货币数据集仅支持数据检索功能，不支持回测功能。*

## 收集器数据


### 加密货币数据

#### 来自 Coingecko 的 1d 数据

```bash

# 从 https://api.coingecko.com/api/v3/ 下载
python collector.py download_data --source_dir ~/.qlib/crypto_data/source/1d --start 2015-01-01 --end 2021-11-30 --delay 1 --interval 1d

# 规范化
python collector.py normalize_data --source_dir ~/.qlib/crypto_data/source/1d --normalize_dir ~/.qlib/crypto_data/source/1d_nor --interval 1d --date_field_name date

# 转储数据
cd qlib/scripts
python dump_bin.py dump_all --data_path ~/.qlib/crypto_data/source/1d_nor --qlib_dir ~/.qlib/qlib_data/crypto_data --freq day --date_field_name date --include_fields prices,total_volumes,market_caps

```

### 使用数据

```python
import qlib
from qlib.data import D

qlib.init(provider_uri="~/.qlib/qlib_data/crypto_data")
df = D.features(D.instruments(market="all"), ["$prices", "$total_volumes","$market_caps"], freq="day")
```


### 帮助
```bash
python collector.py collector_data --help
```

## 参数

- interval: 1d
- delay: 1
