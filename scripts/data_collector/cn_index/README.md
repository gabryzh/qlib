# A股指数（沪深300/中证100/中证500）历史成分股收集

## 环境要求

在运行脚本之前，请先安装所需的依赖包：

```bash
pip install -r requirements.txt
```

## 数据收集命令

本脚本提供了两种功能：

1.  **`parse_instruments`**: 解析指数的完整历史成分股，并生成 `qlib` 所需的 `instruments` 文件。这个文件会记录每只成分股在指数中的加入日期和退出日期。
2.  **`save_new_companies`**: 仅获取当前最新的指数成分股列表。

### 命令示例

```bash
# 功能一：解析沪深300指数的完整历史成分股
# --index_name: 指定指数名称，支持 CSI300, CSI100, CSI500
# --qlib_dir: 指定Qlib数据目录，生成的文件将保存在该目录下的 instruments 子目录中
# --method: 指定要执行的方法
python collector.py get_instruments --index_name CSI300 --qlib_dir ~/.qlib/qlib_data/cn_data --method parse_instruments

# 功能二：仅获取沪深300指数的最新成分股
python collector.py get_instruments --index_name CSI300 --qlib_dir ~/.qlib/qlib_data/cn_data --method save_new_companies

# 查看帮助信息
python collector.py get_instruments --help
```

### 支持的指数

-   `CSI300`: 沪深300
-   `CSI100`: 中证100
-   `CSI500`: 中证500
