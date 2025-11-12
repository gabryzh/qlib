# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.
from pathlib import Path

from setuptools_scm import get_version

try:
    from ._version import version as __version__
except ImportError:
    __version__ = get_version(root="..", relative_to=__file__)
__version__bak = __version__  # 此版本是QlibConfig.reset_qlib_version的备份
import logging
import os
import platform
import re
import subprocess
from typing import Union

from ruamel.yaml import YAML

from .log import get_module_logger


# 初始化 qlib
def init(default_conf="client", **kwargs):
    """

    参数
    ----------
    default_conf: str
        默认值为 client。接受的值：client/server。
    **kwargs :
        clear_mem_cache: str
            默认值为 True；
            是否清除内存缓存。
            当init被多次调用时，通常用于提高性能
        skip_if_reg: bool: str
            默认值为 True；
            使用记录器时，可以将skip_if_reg设置为True以避免记录器丢失。

    """
    from .config import C  # pylint: disable=C0415
    from .data.cache import H  # pylint: disable=C0415

    logger = get_module_logger("Initialization")

    skip_if_reg = kwargs.pop("skip_if_reg", False)
    if skip_if_reg and C.registered:
        # 如果我们在运行实验 `R.start` 期间重新初始化Qlib。
        # 将导致记录器丢失
        logger.warning("由于`skip_if_reg`为True，跳过初始化")
        return

    clear_mem_cache = kwargs.pop("clear_mem_cache", True)
    if clear_mem_cache:
        H.clear()
    C.set(default_conf, **kwargs)
    get_module_logger.setLevel(C.logging_level)

    # 挂载nfs
    for _freq, provider_uri in C.provider_uri.items():
        mount_path = C["mount_path"][_freq]
        # 检查路径是服务器还是本地
        uri_type = C.dpm.get_uri_type(provider_uri)
        if uri_type == C.LOCAL_URI:
            if not Path(provider_uri).exists():
                if C["auto_mount"]:
                    logger.error(
                        f"无效的 provider uri: {provider_uri}，请检查是否设置了有效的 provider uri。此路径不存在。"
                    )
                else:
                    logger.warning(f"auto_path is False，请确保 {mount_path} 已挂载")
        elif uri_type == C.NFS_URI:
            _mount_nfs_uri(provider_uri, C.dpm.get_data_uri(_freq), C["auto_mount"])
        else:
            raise NotImplementedError(f"不支持此类型的URI")

    C.register()

    if "flask_server" in C:
        logger.info(f"flask_server={C['flask_server']}, flask_port={C['flask_port']}")
    logger.info("qlib successfully initialized based on %s settings." % default_conf)
    data_path = {_freq: C.dpm.get_data_uri(_freq) for _freq in C.dpm.provider_uri.keys()}
    logger.info(f"data_path={data_path}")


def _mount_nfs_uri(provider_uri, mount_path, auto_mount: bool = False):
    LOG = get_module_logger("mount nfs", level=logging.INFO)
    if mount_path is None:
        raise ValueError(f"Invalid mount path: {mount_path}!")
    if not re.match(r"^[a-zA-Z0-9.:/\-_]+$", provider_uri):
        raise ValueError(f"Invalid provider_uri format: {provider_uri}")
    # FIXME: C["provider_uri"] 在此函数中被修改
    # 如果它没有被修改，我们只能传递 provider_uri 或 mount_path 而不是 C
    mount_command = ["sudo", "mount.nfs", provider_uri, mount_path]
    # 如果提供程序uri类似于 172.23.233.89//data/csdesign'
    # 它将是一个nfs路径。将使用客户端提供程序
    if not auto_mount:  # pylint: disable=R1702
        if not Path(mount_path).exists():
            raise FileNotFoundError(
                f"无效的挂载路径: {mount_path}! 请手动挂载: {' '.join(mount_command)} 或设置init参数 `auto_mount=True`"
            )
    else:
        # 判断系统类型
        sys_type = platform.system()
        if "windows" in sys_type.lower():
            # 系统: window
            try:
                subprocess.run(
                    ["mount", "-o", "anon", provider_uri, mount_path],
                    capture_output=True,
                    text=True,
                    check=True,
                )
                LOG.info("挂载完成。")
            except subprocess.CalledProcessError as e:
                error_output = (e.stdout or "") + (e.stderr or "")
                if e.returncode == 85:
                    LOG.warning(f"{provider_uri} 已挂载于 {mount_path}")
                elif e.returncode == 53:
                    raise OSError("未找到网络路径") from e
                elif "error" in error_output.lower() or "错误" in error_output:
                    raise OSError("无效的挂载路径") from e
                else:
                    raise OSError(f"未知的挂载错误: {error_output.strip()}") from e
        else:
            # 系统: linux/Unix/Mac
            # 检查挂载
            _remote_uri = provider_uri[:-1] if provider_uri.endswith("/") else provider_uri
            # `mount a /b/c` 不同于 `mount a /b/c/`。因此我们将其转换为字符串以确保准确处理
            mount_path = str(mount_path)
            _mount_path = mount_path[:-1] if mount_path.endswith("/") else mount_path
            _check_level_num = 2
            _is_mount = False
            while _check_level_num:
                with subprocess.Popen(
                    ["mount"],
                    text=True,
                    stdout=subprocess.PIPE,
                    stderr=subprocess.STDOUT,
                ) as shell_r:
                    _command_log = shell_r.stdout.readlines()
                    _command_log = [line for line in _command_log if _remote_uri in line]
                if len(_command_log) > 0:
                    for _c in _command_log:
                        if isinstance(_c, str):
                            _temp_mount = _c.split(" ")[2]
                        else:
                            _temp_mount = _c.decode("utf-8").split(" ")[2]
                        _temp_mount = _temp_mount[:-1] if _temp_mount.endswith("/") else _temp_mount
                        if _temp_mount == _mount_path:
                            _is_mount = True
                            break
                if _is_mount:
                    break
                _remote_uri = "/".join(_remote_uri.split("/")[:-1])
                _mount_path = "/".join(_mount_path.split("/")[:-1])
                _check_level_num -= 1

            if not _is_mount:
                try:
                    Path(mount_path).mkdir(parents=True, exist_ok=True)
                except Exception as e:
                    raise OSError(
                        f"创建目录 {mount_path} 失败，请手动创建 {mount_path}！"
                    ) from e

                # 检查 nfs-common
                command_res = os.popen("dpkg -l | grep nfs-common")
                command_res = command_res.readlines()
                if not command_res:
                    raise OSError("未找到 nfs-common，请执行以下命令安装：sudo apt install nfs-common")
                # 手动挂载
                try:
                    subprocess.run(mount_command, check=True, capture_output=True, text=True)
                    LOG.info("挂载完成。")
                except subprocess.CalledProcessError as e:
                    if e.returncode == 256:
                        raise OSError("挂载失败：需要sudo或权限被拒绝") from e
                    elif e.returncode == 32512:
                        raise OSError(f"挂载 {provider_uri} 到 {mount_path} 出错！命令错误") from e
                    else:
                        raise OSError(f"挂载失败：{e.stderr}") from e
            else:
                LOG.warning(f"{_remote_uri} on {_mount_path} 已挂载")


def init_from_yaml_conf(conf_path, **kwargs):
    """从yaml配置文件初始化

    :param conf_path: yml格式的qlib配置文件路径
    """

    if conf_path is None:
        config = {}
    else:
        with open(conf_path) as f:
            yaml = YAML(typ="safe", pure=True)
            config = yaml.load(f)
    config.update(kwargs)
    default_conf = config.pop("default_conf", "client")
    init(default_conf, **config)


def get_project_path(config_name="config.yaml", cur_path: Union[Path, str, None] = None) -> Path:
    """
    如果用户正在构建遵循以下模式的项目。
    - Qlib是项目路径中的一个子文件夹
    - qlib中有一个名为`config.yaml`的文件。

    例如:
        如果您的项目文件系统结构遵循这样的模式

            <project_path>/
              - config.yaml
              - ...一些文件夹...
                - qlib/

        此文件夹将返回<project_path>

        注意：此处不支持链接。


    此方法通常在以下情况下使用
    - 用户希望使用相对配置路径，而不是在代码中硬编码qlib配置路径

    引发
    ------
    FileNotFoundError:
        如果找不到项目路径
    """
    if cur_path is None:
        cur_path = Path(__file__).absolute().resolve()
    cur_path = Path(cur_path)
    while True:
        if (cur_path / config_name).exists():
            return cur_path
        if cur_path == cur_path.parent:
            raise FileNotFoundError("We can't find the project path")
        cur_path = cur_path.parent


def auto_init(**kwargs):
    """
    此函数将按以下优先级自动初始化qlib
    - 查找项目配置并初始化qlib
        - 解析过程将受配置文件`conf_type`的影响
    - 使用默认配置初始化qlib
    - 如果已初始化，则跳过初始化

    :**kwargs: 可能包含以下参数
                cur_path: 查找项目路径的起始路径

    以下是配置的两个示例

    示例1)
    如果要基于共享配置创建新的项目特定配置，可以使用 `conf_type: ref`

    .. code-block:: yaml

        conf_type: ref
        qlib_cfg: '<shared_yaml_config_path>'    # 这可以是空引用，没有来自其他文件的配置
        # `qlib_cfg_update`中的以下配置是特定于项目的
        qlib_cfg_update:
            exp_manager:
                class: "MLflowExpManager"
                module_path: "qlib.workflow.expm"
                kwargs:
                    uri: "file://<your mlflow experiment path>"
                    default_exp_name: "Experiment"

    示例2)
    如果要创建简单的独立配置，可以使用以下配置(又名 `conf_type: origin`)

    .. code-block:: python

        exp_manager:
            class: "MLflowExpManager"
            module_path: "qlib.workflow.expm"
            kwargs:
                uri: "file://<your mlflow experiment path>"
                default_exp_name: "Experiment"

    """
    kwargs["skip_if_reg"] = kwargs.get("skip_if_reg", True)

    try:
        pp = get_project_path(cur_path=kwargs.pop("cur_path", None))
    except FileNotFoundError:
        init(**kwargs)
    else:
        logger = get_module_logger("Initialization")
        conf_pp = pp / "config.yaml"
        with conf_pp.open() as f:
            yaml = YAML(typ="safe", pure=True)
            conf = yaml.load(f)

        conf_type = conf.get("conf_type", "origin")
        if conf_type == "origin":
            # The type of config is just like original qlib config
            init_from_yaml_conf(conf_pp, **kwargs)
        elif conf_type == "ref":
            # This config type will be more convenient in following scenario
            # - There is a shared configure file, and you don't want to edit it inplace.
            # - The shared configure may be updated later, and you don't want to copy it.
            # - You have some customized config.
            qlib_conf_path = conf.get("qlib_cfg", None)

            # merge the arguments
            qlib_conf_update = conf.get("qlib_cfg_update", {})
            for k, v in kwargs.items():
                if k in qlib_conf_update:
                    logger.warning(f"`qlib_conf_update` from conf_pp is override by `kwargs` on key '{k}'")
            qlib_conf_update.update(kwargs)

            init_from_yaml_conf(qlib_conf_path, **qlib_conf_update)
        logger.info(f"Auto load project config: {conf_pp}")
