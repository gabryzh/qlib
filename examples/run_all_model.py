#  Copyright (c) Microsoft Corporation.
#  Licensed under the MIT License.

import os
import sys
import fire
import time
import glob
import shutil
import signal
import inspect
import tempfile
import functools
import statistics
import subprocess
from datetime import datetime
from ruamel.yaml import YAML
from pathlib import Path
from operator import xor
from pprint import pprint

import qlib
from qlib.workflow import R
from qlib.tests.data import GetData


# 装饰器：检查参数
def only_allow_defined_args(function_to_decorate):
    """
    一个装饰器，用于确保函数只接收在其定义中声明过的参数。
    """
    @functools.wraps(function_to_decorate)
    def _return_wrapped(*args, **kwargs):
        """内部包装函数。"""
        # 获取被装饰函数的参数规格
        argspec = inspect.getfullargspec(function_to_decorate)
        # 创建一个包含所有有效参数名的集合
        valid_names = set(argspec.args + argspec.kwonlyargs)
        if "self" in valid_names:
            valid_names.remove("self")
        # 遍历传入的关键字参数
        for arg_name in kwargs:
            # 如果参数名无效，则抛出 ValueError
            if arg_name not in valid_names:
                raise ValueError("未知参数 '%s'，预期参数: [%s]" % (arg_name, ", ".join(valid_names)))
        # 如果所有参数都有效，则调用原始函数
        return function_to_decorate(*args, **kwargs)

    return _return_wrapped


# 函数：处理 ctrl+z 和 ctrl+c
def handler(signum, frame):
    """
    信号处理函数，用于捕获中断信号 (SIGINT) 并终止当前进程。
    """
    os.system("kill -9 %d" % os.getpid())


# 注册信号处理函数
signal.signal(signal.SIGINT, handler)


# 函数：计算结果字典中列表的均值和标准差
def cal_mean_std(results) -> dict:
    """
    计算一个字典中每个指标列表的均值和标准差。

    :param results: 一个字典，键为模型名称，值为包含指标列表的另一个字典。
    :return: 一个包含每个指标均值和标准差的字典。
    """
    mean_std = dict()
    for fn in results:
        mean_std[fn] = dict()
        for metric in results[fn]:
            # 计算均值
            mean = statistics.mean(results[fn][metric]) if len(results[fn][metric]) > 1 else results[fn][metric][0]
            # 计算标准差
            std = statistics.stdev(results[fn][metric]) if len(results[fn][metric]) > 1 else 0
            mean_std[fn][metric] = [mean, std]
    return mean_std


# 函数：为 anaconda 环境创建环境
def create_env():
    """
    创建一个临时的 anaconda 虚拟环境。

    :return: 临时目录路径, 环境路径, python 解释器路径, anaconda 激活脚本路径
    """
    # 创建临时目录
    temp_dir = tempfile.mkdtemp()
    env_path = Path(temp_dir).absolute()
    sys.stderr.write(f"正在创建虚拟环境，路径: {env_path}...\n")
    # 创建 conda 环境
    execute(f"conda create --prefix {env_path} python=3.7 -y")
    python_path = env_path / "bin" / "python"  # TODO: 修复此处的硬编码路径
    sys.stderr.write("\n")
    # 获取 anaconda 激活脚本路径
    conda_activate = Path(os.environ["CONDA_PREFIX"]) / "bin" / "activate"  # TODO: 修复此处的硬编码路径
    return temp_dir, env_path, python_path, conda_activate


# 函数：执行命令行命令
def execute(cmd, wait_when_err=False, raise_err=True):
    """
    执行一个 shell 命令。

    :param cmd: 要执行的命令。
    :param wait_when_err: 如果发生错误，是否等待用户输入。
    :param raise_err: 如果发生错误，是否抛出异常。
    :return: 如果命令执行成功，返回 None；否则返回错误信息。
    """
    print("正在运行命令:", cmd)
    with subprocess.Popen(cmd, stdout=subprocess.PIPE, bufsize=1, universal_newlines=True, shell=True) as p:
        for line in p.stdout:
            sys.stdout.write(line.split("\b")[0])
            if "\b" in line:
                sys.stdout.flush()
                time.sleep(0.1)
                sys.stdout.write("\b" * 10 + "\b".join(line.split("\b")[1:-1]))

    if p.returncode != 0:
        if wait_when_err:
            input("按回车键继续")
        if raise_err:
            raise RuntimeError(f"执行命令时出错: {cmd}")
        return p.stderr
    else:
        return None


# 函数：获取 benchmarks 文件夹下的所有文件夹
def get_all_folders(models, exclude) -> dict:
    """
    获取 `benchmarks` 目录下的所有模型文件夹。

    :param models: 要包含或排除的模型列表。
    :param exclude: 一个布尔值，如果为 True，则排除 `models` 中指定的模型；否则只包含 `models` 中指定的模型。
    :return: 一个包含模型名称和对应路径的字典。
    """
    folders = dict()
    # 处理 models 参数
    if isinstance(models, str):
        model_list = models.split(",")
        models = [m.lower().strip("[ ]") for m in model_list]
    elif isinstance(models, list):
        models = [m.lower() for m in models]
    elif models is None:
        models = [f.name.lower() for f in os.scandir("benchmarks")]
    else:
        raise ValueError("输入的 models 类型不支持。请提供字符串或列表，不要带空格。")

    # 遍历 benchmarks 目录下的所有文件夹
    for f in os.scandir("benchmarks"):
        add = xor(bool(f.name.lower() in models), bool(exclude))
        if add:
            path = Path("benchmarks") / f.name
            folders[f.name] = str(path.resolve())
    return folders


# 函数：获取模型文件夹下的所有文件
def get_all_files(folder_path, dataset, universe="") -> (str, str):
    """
    获取指定模型文件夹下的 yaml 配置文件和 requirements.txt 文件。

    :param folder_path: 模型文件夹的路径。
    :param dataset: 数据集名称。
    :param universe: 股票领域。
    :return: yaml 文件路径和 requirements.txt 文件路径。
    """
    if universe != "":
        universe = f"_{universe}"
    yaml_path = str(Path(f"{folder_path}") / f"*{dataset}{universe}.yaml")
    req_path = str(Path(f"{folder_path}") / f"*.txt")
    yaml_file = glob.glob(yaml_path)
    req_file = glob.glob(req_path)
    if len(yaml_file) == 0:
        return None, None
    else:
        return yaml_file[0], req_file[0]


# 函数：检索所有结果
def get_all_results(folders) -> dict:
    """
    从 MLflow 实验中检索所有模型的结果。

    :param folders: 包含模型名称的列表。
    :return: 一个包含所有模型结果的字典。
    """
    results = dict()
    for fn in folders:
        try:
            exp = R.get_exp(experiment_name=fn, create=False)
        except ValueError:
            # 没有实验结果
            continue
        recorders = exp.list_recorders()
        result = dict()
        # 初始化指标列表
        result["annualized_return_with_cost"] = list()
        result["information_ratio_with_cost"] = list()
        result["max_drawdown_with_cost"] = list()
        result["ic"] = list()
        result["icir"] = list()
        result["rank_ic"] = list()
        result["rank_icir"] = list()
        # 遍历所有记录器
        for recorder_id in recorders:
            if recorders[recorder_id].status == "FINISHED":
                recorder = R.get_recorder(recorder_id=recorder_id, experiment_name=fn)
                metrics = recorder.list_metrics()
                if "1day.excess_return_with_cost.annualized_return" not in metrics:
                    print(f"{recorder_id} 由于结果不完整而被跳过")
                    continue
                # 提取指标
                result["annualized_return_with_cost"].append(metrics["1day.excess_return_with_cost.annualized_return"])
                result["information_ratio_with_cost"].append(metrics["1day.excess_return_with_cost.information_ratio"])
                result["max_drawdown_with_cost"].append(metrics["1day.excess_return_with_cost.max_drawdown"])
                result["ic"].append(metrics["IC"])
                result["icir"].append(metrics["ICIR"])
                result["rank_ic"].append(metrics["Rank IC"])
                result["rank_icir"].append(metrics["Rank ICIR"])
        results[fn] = result
    return results


# 函数：生成并保存 markdown 表格
def gen_and_save_md_table(metrics, dataset):
    """
    生成一个 markdown 表格来展示模型结果，并将其保存到文件中。

    :param metrics: 包含模型指标的字典。
    :param dataset: 数据集名称。
    :return: 生成的 markdown 表格字符串。
    """
    table = "| Model Name | Dataset | IC | ICIR | Rank IC | Rank ICIR | Annualized Return | Information Ratio | Max Drawdown |\n"
    table += "|---|---|---|---|---|---|---|---|---|\n"
    for fn in metrics:
        ic = metrics[fn]["ic"]
        icir = metrics[fn]["icir"]
        ric = metrics[fn]["rank_ic"]
        ricir = metrics[fn]["rank_icir"]
        ar = metrics[fn]["annualized_return_with_cost"]
        ir = metrics[fn]["information_ratio_with_cost"]
        md = metrics[fn]["max_drawdown_with_cost"]
        table += f"| {fn} | {dataset} | {ic[0]:5.4f}±{ic[1]:2.2f} | {icir[0]:5.4f}±{icir[1]:2.2f}| {ric[0]:5.4f}±{ric[1]:2.2f} | {ricir[0]:5.4f}±{ricir[1]:2.2f} | {ar[0]:5.4f}±{ar[1]:2.2f} | {ir[0]:5.4f}±{ir[1]:2.2f}| {md[0]:5.4f}±{md[1]:2.2f} |\n"
    pprint(table)
    with open("table.md", "w") as f:
        f.write(table)
    return table


# 读取 yaml，删除模型的 seed 参数，然后将文件保存在临时目录中
def gen_yaml_file_without_seed_kwargs(yaml_path, temp_dir):
    """
    读取 yaml 配置文件，删除模型中的 `seed` 参数，并将修改后的配置保存到临时目录中。
    这对于多次运行模型以获得更稳健的结果非常有用。

    :param yaml_path: 原始 yaml 文件的路径。
    :param temp_dir: 用于保存修改后 yaml 文件的临时目录。
    :return: 修改后 yaml 文件的路径。
    """
    with open(yaml_path, "r") as fp:
        yaml = YAML(typ="safe", pure=True)
        config = yaml.load(fp)
    try:
        del config["task"]["model"]["kwargs"]["seed"]
    except KeyError:
        # 如果关键字不存在，则使用原始 yaml
        # 注意：如果模型必须在原始路径中运行（当使用 sys.rel_path 时），这一点非常重要
        return yaml_path
    else:
        # 否则，生成一个没有随机种子的新 yaml
        file_name = yaml_path.split("/")[-1]
        temp_path = os.path.join(temp_dir, file_name)
        with open(temp_path, "w") as fp:
            yaml.dump(config, fp)
        return temp_path


class ModelRunner:
    """
    一个用于运行 `qlib` 模型的类。
    """
    def _init_qlib(self, exp_folder_name):
        """
        初始化 `qlib`。

        :param exp_folder_name: 实验文件夹的名称。
        """
        # 初始化 qlib
        GetData().qlib_data(exists_skip=True)
        qlib.init(
            exp_manager={
                "class": "MLflowExpManager",
                "module_path": "qlib.workflow.expm",
                "kwargs": {
                    "uri": "file:" + str(Path(os.getcwd()).resolve() / exp_folder_name),
                    "default_exp_name": "Experiment",
                },
            }
        )

    # 函数：运行所有模型
    @only_allow_defined_args
    def run(
        self,
        times=1,
        models=None,
        dataset="Alpha360",
        universe="",
        exclude=False,
        qlib_uri: str = "git+https://github.com/microsoft/qlib#egg=pyqlib",
        exp_folder_name: str = "run_all_model_records",
        wait_before_rm_env: bool = False,
        wait_when_err: bool = False,
    ):
        """
        请注意，此函数只能在 Linux 下工作。将来会支持 MacOS 和 Windows。
        非常欢迎任何有助于增强此方法的 PR。此外，此脚本不支持并行多次运行同一模型，
        这将在未来的开发中修复。

        参数:
        -----------
        times : int
            确定模型应该运行多少次。
        models : str or list
            确定要运行或排除的特定模型或模型列表。
        exclude : boolean
            确定是排除还是包含正在使用的模型。
        dataset : str
            确定用于每个模型的数据集。
        universe : str
            数据集的股票领域。
            默认 "" 表示
        qlib_uri : str
            用 pip 安装 qlib 的 uri
            可以是远程 URI 或本地路径（注意：本地路径必须是绝对路径）
        exp_folder_name: str
            实验文件夹的名称
        wait_before_rm_env : bool
            在删除环境之前等待。
        wait_when_err : bool
            在执行命令时出现错误时等待

        用法:
        -------
        以下是该函数在 bash 中的一些用例：

        run_all_models 将根据 `models` `dataset` `universe` 决定运行哪个配置
        示例 1):

            models="lightgbm", dataset="Alpha158", universe="" 将导致运行以下配置
            examples/benchmarks/LightGBM/workflow_config_lightgbm_Alpha158.yaml

            models="lightgbm", dataset="Alpha158", universe="csi500" 将导致运行以下配置
            examples/benchmarks/LightGBM/workflow_config_lightgbm_Alpha158_csi500.yaml

        .. code-block:: bash

            # 情况 1 - 多次运行所有模型
            python run_all_model.py run 3

            # 情况 2 - 多次运行特定模型
            python run_all_model.py run 3 mlp

            # 情况 3 - 使用特定数据集多次运行特定模型
            python run_all_model.py run 3 mlp Alpha158

            # 情况 4 - 运行除作为参数给出的模型之外的其他模型多次
            python run_all_model.py run 3 [mlp,tft,lstm] --exclude=True

            # 情况 5 - 运行特定模型一次
            python run_all_model.py run --models=[mlp,lightgbm]

            # 情况 6 - 运行除作为参数给出的模型之外的其他模型一次
            python run_all_model.py run --models=[mlp,tft,sfm] --exclude=True

            # 情况 7 - 在 csi500 上运行 lightgbm 模型。
            python run_all_model.py run 3 lightgbm Alpha158 csi500

        """
        self._init_qlib(exp_folder_name)

        # 获取所有文件夹
        folders = get_all_folders(models, exclude)
        # 初始化错误消息：
        errors = dict()
        # 迭代运行所有模型
        for fn in folders:
            # 获取所有文件
            sys.stderr.write("正在检索文件...\n")
            yaml_path, req_path = get_all_files(folders[fn], dataset, universe=universe)
            if yaml_path is None:
                sys.stderr.write(f"在 {folders[fn]} 中没有 {dataset}.yaml 文件")
                continue
            sys.stderr.write("\n")
            # 通过 anaconda 创建环境
            temp_dir, env_path, python_path, conda_activate = create_env()

            # 安装 requirements.txt
            sys.stderr.write("正在安装 requirements.txt...\n")
            with open(req_path) as f:
                content = f.read()
            if "torch" in content:
                # 根据 nvidia 版本自动安装 pytorch
                execute(
                    f"{python_path} -m pip install light-the-torch", wait_when_err=wait_when_err
                )  # 用于根据 nvidia 驱动程序自动安装 torch
                execute(
                    f"{env_path / 'bin' / 'ltt'} install --install-cmd '{python_path} -m pip install {{packages}}' -- -r {req_path}",
                    wait_when_err=wait_when_err,
                )
            else:
                execute(f"{python_path} -m pip install -r {req_path}", wait_when_err=wait_when_err)
            sys.stderr.write("\n")

            # 读取 yaml，删除模型的 seed 参数，然后将文件保存在临时目录中
            yaml_path = gen_yaml_file_without_seed_kwargs(yaml_path, temp_dir)
            # 为 tft 设置 gpu
            if fn == "TFT":
                execute(
                    f"conda install -y --prefix {env_path} anaconda cudatoolkit=10.0 && conda install -y --prefix {env_path} cudnn",
                    wait_when_err=wait_when_err,
                )
                sys.stderr.write("\n")
            # 安装 qlib
            sys.stderr.write("正在安装 qlib...\n")
            execute(f"{python_path} -m pip install --upgrade pip", wait_when_err=wait_when_err)  # TODO: 修复此处的硬编码路径
            execute(f"{python_path} -m pip install --upgrade cython", wait_when_err=wait_when_err)  # TODO: 修复此处的硬编码路径
            if fn == "TFT":
                execute(
                    f"cd {env_path} && {python_path} -m pip install --upgrade --force-reinstall --ignore-installed PyYAML -e {qlib_uri}",
                    wait_when_err=wait_when_err,
                )  # TODO: 修复此处的硬编码路径
            else:
                execute(
                    f"cd {env_path} && {python_path} -m pip install --upgrade --force-reinstall -e {qlib_uri}",
                    wait_when_err=wait_when_err,
                )  # TODO: 修复此处的硬编码路径
            sys.stderr.write("\n")
            # 多次运行 workflow_by_config
            for i in range(times):
                sys.stderr.write(f"正在运行模型: {fn}，第 {i+1} 次迭代...\n")
                errs = execute(
                    f"{python_path} {env_path / 'bin' / 'qrun'} {yaml_path} {fn} {exp_folder_name}",
                    wait_when_err=wait_when_err,
                )
                if errs is not None:
                    _errs = errors.get(fn, {})
                    _errs.update({i: errs})
                    errors[fn] = _errs
                sys.stderr.write("\n")
            # 删除环境
            sys.stderr.write(f"正在删除环境: {env_path}...\n")
            if wait_before_rm_env:
                input("按回车键继续")
            shutil.rmtree(env_path)
        # 打印错误
        sys.stderr.write(f"以下是模型的一些错误...\n")
        pprint(errors)
        self._collect_results(exp_folder_name, dataset)

    def _collect_results(self, exp_folder_name, dataset):
        """
        收集、处理并保存所有模型的结果。
        """
        folders = get_all_folders(exp_folder_name, dataset)
        # 获取所有结果
        sys.stderr.write(f"正在检索结果...\n")
        results = get_all_results(folders)
        if len(results) > 0:
            # 计算均值和标准差
            sys.stderr.write(f"正在计算结果的均值和标准差...\n")
            results = cal_mean_std(results)
            # 生成 md 表格
            sys.stderr.write(f"正在生成 markdown 表格...\n")
            gen_and_save_md_table(results, dataset)
            sys.stderr.write("\n")
        sys.stderr.write("\n")
        # 移动结果文件夹
        shutil.move(exp_folder_name, exp_folder_name + f"_{dataset}_{datetime.now().strftime('%Y-%m-%d_%H:%M:%S')}")
        shutil.move("table.md", f"table_{dataset}_{datetime.now().strftime('%Y-%m-%d_%H:%M:%S')}.md")


if __name__ == "__main__":
    fire.Fire(ModelRunner)  # 运行所有模型
