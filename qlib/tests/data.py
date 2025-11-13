# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

import os
import re
import sys
import qlib
import shutil
import zipfile
import requests
import datetime
from tqdm import tqdm
from pathlib import Path
from loguru import logger
from qlib.utils import exists_qlib_data


class GetData:
    """获取数据的类"""
    REMOTE_URL = "https://github.com/SunsetWolf/qlib_dataset/releases/download"

    def __init__(self, delete_zip_file=False):
        """

        Parameters
        ----------
        delete_zip_file : bool, optional
            是否删除zip文件，值为True或False，默认为False
        """
        self.delete_zip_file = delete_zip_file

    def merge_remote_url(self, file_name: str):
        """
        生成下载链接。

        Parameters
        ----------
        file_name: str
            要下载的文件名。
            文件名可以附带版本号，（例如：v2/qlib_data_simple_cn_1d_latest.zip），
            如果没有附加版本号，默认从v0版本下载。
        """
        return f"{self.REMOTE_URL}/{file_name}" if "/" in file_name else f"{self.REMOTE_URL}/v0/{file_name}"

    def download(self, url: str, target_path: [Path, str]):
        """
        从指定url下载文件。

        Parameters
        ----------
        url: str
            数据的url。
        target_path: str
            数据保存的位置，包括文件名。
        """
        file_name = str(target_path).rsplit("/", maxsplit=1)[-1]
        resp = requests.get(url, stream=True, timeout=60)
        resp.raise_for_status()
        if resp.status_code != 200:
            raise requests.exceptions.HTTPError()

        chunk_size = 1024
        logger.warning(
            f"The data for the example is collected from Yahoo Finance. Please be aware that the quality of the data might not be perfect. (You can refer to the original data source: https://finance.yahoo.com/lookup.)"
        )
        logger.info(f"{os.path.basename(file_name)} downloading......")
        with tqdm(total=int(resp.headers.get("Content-Length", 0))) as p_bar:
            with target_path.open("wb") as fp:
                for chunk in resp.iter_content(chunk_size=chunk_size):
                    fp.write(chunk)
                    p_bar.update(chunk_size)

    def download_data(self, file_name: str, target_dir: [Path, str], delete_old: bool = True):
        """
        将指定的文件下载到目标文件夹。

        Parameters
        ----------
        target_dir: str
            数据保存目录
        file_name: str
            数据集名称，需要以.zip结尾，值来自[rl_data.zip, csv_data_cn.zip, ...]
            可能包含文件夹名称，例如：v2/qlib_data_simple_cn_1d_latest.zip
        delete_old: bool
            删除现有目录，默认为True

        Examples
        ---------
        # get rl data
        python get_data.py download_data --file_name rl_data.zip --target_dir ~/.qlib/qlib_data/rl_data
        When this command is run, the data will be downloaded from this link: https://qlibpublic.blob.core.windows.net/data/default/stock_data/rl_data.zip?{token}

        # get cn csv data
        python get_data.py download_data --file_name csv_data_cn.zip --target_dir ~/.qlib/csv_data/cn_data
        When this command is run, the data will be downloaded from this link: https://qlibpublic.blob.core.windows.net/data/default/stock_data/csv_data_cn.zip?{token}
        -------

        """
        target_dir = Path(target_dir).expanduser()
        target_dir.mkdir(exist_ok=True, parents=True)
        # 保存的文件名
        _target_file_name = datetime.datetime.now().strftime("%Y%m%d%H%M%S") + "_" + os.path.basename(file_name)
        target_path = target_dir.joinpath(_target_file_name)

        url = self.merge_remote_url(file_name)
        self.download(url=url, target_path=target_path)

        self._unzip(target_path, target_dir, delete_old)
        if self.delete_zip_file:
            target_path.unlink()

    def check_dataset(self, file_name: str):
        url = self.merge_remote_url(file_name)
        resp = requests.get(url, stream=True, timeout=60)
        status = True
        if resp.status_code == 404:
            status = False
        return status

    @staticmethod
    def _unzip(file_path: [Path, str], target_dir: [Path, str], delete_old: bool = True):
        file_path = Path(file_path)
        target_dir = Path(target_dir)
        if delete_old:
            logger.warning(
                f"will delete the old qlib data directory(features, instruments, calendars, features_cache, dataset_cache): {target_dir}"
            )
            GetData._delete_qlib_data(target_dir)
        logger.info(f"{file_path} unzipping......")
        with zipfile.ZipFile(str(file_path.resolve()), "r") as zp:
            for _file in tqdm(zp.namelist()):
                zp.extract(_file, str(target_dir.resolve()))

    @staticmethod
    def _delete_qlib_data(file_dir: Path):
        rm_dirs = []
        for _name in ["features", "calendars", "instruments", "features_cache", "dataset_cache"]:
            _p = file_dir.joinpath(_name)
            if _p.exists():
                rm_dirs.append(str(_p.resolve()))
        if rm_dirs:
            flag = input(
                f"Will be deleted: "
                f"\n\t{rm_dirs}"
                f"\nIf you do not need to delete {file_dir}, please change the <--target_dir>"
                f"\nAre you sure you want to delete, yes(Y/y), no (N/n):"
            )
            if str(flag) not in ["Y", "y"]:
                sys.exit()
            for _p in rm_dirs:
                logger.warning(f"delete: {_p}")
                shutil.rmtree(_p)

    def qlib_data(
        self,
        name="qlib_data",
        target_dir="~/.qlib/qlib_data/cn_data",
        version=None,
        interval="1d",
        region="cn",
        delete_old=True,
        exists_skip=False,
    ):
        """从远程下载cn qlib数据

        Parameters
        ----------
        target_dir: str
            数据保存目录
        name: str
            数据集名称，值来自[qlib_data, qlib_data_simple]，默认为qlib_data
        version: str
            数据版本，值来自[v1, ...]，默认为None(使用脚本指定版本)
        interval: str
            数据频率，值来自[1d]，默认为1d
        region: str
            数据区域，值来自[cn, us]，默认为cn
        delete_old: bool
            删除现有目录，默认为True
        exists_skip: bool
            存在时跳过，默认为False

        Examples
        ---------
        # get 1d data
        python get_data.py qlib_data --name qlib_data --target_dir ~/.qlib/qlib_data/cn_data --interval 1d --region cn
        When this command is run, the data will be downloaded from this link: https://qlibpublic.blob.core.windows.net/data/default/stock_data/v2/qlib_data_cn_1d_latest.zip?{token}

        # get 1min data
        python get_data.py qlib_data --name qlib_data --target_dir ~/.qlib/qlib_data/cn_data_1min --interval 1min --region cn
        When this command is run, the data will be downloaded from this link: https://qlibpublic.blob.core.windows.net/data/default/stock_data/v2/qlib_data_cn_1min_latest.zip?{token}
        -------

        """
        if exists_skip and exists_qlib_data(target_dir):
            logger.warning(
                f"Data already exists: {target_dir}, the data download will be skipped\n"
                f"\tIf downloading is required: `exists_skip=False` or `change target_dir`"
            )
            return

        qlib_version = ".".join(re.findall(r"(\d+)\.+", qlib.__version__))

        def _get_file_name_with_version(qlib_version, dataset_version):
            dataset_version = "v2" if dataset_version is None else dataset_version
            file_name_with_version = f"{dataset_version}/{name}_{region.lower()}_{interval.lower()}_{qlib_version}.zip"
            return file_name_with_version

        file_name = _get_file_name_with_version(qlib_version, dataset_version=version)
        if not self.check_dataset(file_name):
            file_name = _get_file_name_with_version("latest", dataset_version=version)
        self.download_data(file_name.lower(), target_dir, delete_old)
