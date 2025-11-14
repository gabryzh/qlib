# 众包数据

## 初衷
像雅虎这样的公共数据源存在缺陷，它可能会丢失已退市股票的数据，也可能包含错误数据。这会将幸存者偏差引入我们的训练过程。

引入众包数据是为了合并来自多个数据源的数据并相互交叉验证，以便：
1. 我们将拥有更完整的历史记录。
2. 我们可以识别异常数据并在必要时进行修正。

## 相关仓库
原始数据托管在 dolthub 仓库：https://www.dolthub.com/repositories/chenditc/investment_data

处理脚本和 sql 托管在 github 仓库：https://github.com/chenditc/investment_data

打包的 docker 运行时托管在 dockerhub：https://hub.docker.com/repository/docker/chenditc/investment_data

## 如何在 qlib 中使用
### 选项 1：下载发布的二进制数据
用户可以下载 qlib 二进制格式的数据并直接使用：https://github.com/chenditc/investment_data/releases/latest
```bash
wget https://github.com/chenditc/investment_data/releases/latest/download/qlib_bin.tar.gz
tar -zxvf qlib_bin.tar.gz -C ~/.qlib/qlib_data/cn_data --strip-components=2
```

### 选项 2：从 dolthub 生成 qlib 数据
Dolthub 数据将每日更新，因此如果用户想获取最新数据，他们可以使用 docker 转储 qlib 二进制文件：
```
docker run -v /<某个输出目录>:/output -it --rm chenditc/investment_data bash dump_qlib_bin.sh && cp ./qlib_bin.tar.gz /output/
```

## 常见问题及其他信息
见：https://github.com/chenditc/investment_data/blob/main/README.md
