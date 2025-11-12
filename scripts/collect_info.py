# 导入必要的库
import sys  # 用于获取Python环境信息
import platform  # 用于获取操作系统信息
import qlib  # 导入qlib以获取其版本号
import fire  # 用于创建命令行界面
import pkg_resources  # 用于获取已安装包的版本信息
from pathlib import Path  # 用于处理文件路径

# 定义qlib项目的根目录路径
QLIB_PATH = Path(__file__).absolute().resolve().parent.parent


class InfoCollector:
    """
    用户可以通过以下命令收集系统信息：
    `cd scripts && python collect_info.py all`
    - 注意：请避免在包含`qlib`的项目文件夹中运行此脚本。
    """

    def sys(self):
        """收集与系统相关的信息"""
        # 遍历platform模块中的方法，打印系统、机器、平台和版本信息
        for method in ["system", "machine", "platform", "version"]:
            print(getattr(platform, method)())

    def py(self):
        """收集与Python相关的信息"""
        # 打印Python的版本信息，并将换行符替换为空格
        print("Python version: {}".format(sys.version.replace("\n", " ")))

    def qlib(self):
        """收集与qlib相关的信息"""
        # 打印qlib的版本
        print("Qlib version: {}".format(qlib.__version__))

        # 定义qlib的核心依赖列表
        REQUIRED = [
            "setuptools",
            "wheel",
            "cython",
            "pyyaml",
            "numpy",
            "pandas",
            "mlhost",
            "filelock",
            "redis",
            "dill",
            "fire",
            "ruamel.yaml",
            "python-redis-lock",
            "tqdm",
            "pymongo",
            "loguru",
            "lightgbm",
            "gym",
            "cvxpy",
            "joblib",
            "matplotlib",
            "jupyter",
            "nbconvert",
            "pyarrow",
            "pydantic-settings",
            "setuptools-scm",
        ]

        # 遍历依赖列表，获取并打印每个包的版本号
        for package in REQUIRED:
            try:
                version = pkg_resources.get_distribution(package).version
                print(f"{package}=={version}")
            except pkg_resources.DistributionNotFound:
                print(f"{package} is not installed.")


    def all(self):
        """收集所有信息"""
        # 依次调用sys, py, qlib方法，打印所有信息
        for method in ["sys", "py", "qlib"]:
            getattr(self, method)()
            print()  # 打印一个空行以分隔不同部分的信息


if __name__ == "__main__":
    # 使用fire库将InfoCollector类暴露为命令行工具
    # 例如，可以在命令行中运行 `python collect_info.py all`
    fire.Fire(InfoCollector)
