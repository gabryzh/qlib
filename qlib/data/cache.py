# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.


from __future__ import division
from __future__ import print_function

import os
import sys
import stat
import time
import pickle
import traceback
import redis_lock
import contextlib
import abc
from pathlib import Path
import numpy as np
import pandas as pd
from typing import Union, Iterable
from collections import OrderedDict

from ..config import C
from ..utils import (
    hash_args,
    get_redis_connection,
    read_bin,
    parse_field,
    remove_fields_space,
    normalize_cache_fields,
    normalize_cache_instruments,
)

from ..log import get_module_logger
from .base import Feature
from .ops import Operators  # pylint: disable=W0611  # noqa: F401


class QlibCacheException(RuntimeError):
    """Qlib 缓存异常类"""
    pass


class MemCacheUnit(abc.ABC):
    """内存缓存单元的基类"""

    def __init__(self, *args, **kwargs):
        self.size_limit = kwargs.pop("size_limit", 0)
        self._size = 0
        self.od = OrderedDict()

    def __setitem__(self, key, value):
        # TODO: 线程安全？__setitem__ 失败可能会导致大小不一致？

        # 预计算 od.__setitem__ 之后的大小
        self._adjust_size(key, value)

        self.od.__setitem__(key, value)

        # 将键移动到末尾，使其成为最新的
        self.od.move_to_end(key)

        if self.limited:
            # 弹出超出大小限制的最旧项目
            while self._size > self.size_limit:
                self.popitem(last=False)

    def __getitem__(self, key):
        v = self.od.__getitem__(key)
        self.od.move_to_end(key)
        return v

    def __contains__(self, key):
        return key in self.od

    def __len__(self):
        return self.od.__len__()

    def __repr__(self):
        return f"{self.__class__.__name__}<size_limit:{self.size_limit if self.limited else 'no limit'} total_size:{self._size}>\n{self.od.__repr__()}"

    def set_limit_size(self, limit):
        self.size_limit = limit

    @property
    def limited(self):
        """内存缓存是否受限"""
        return self.size_limit > 0

    @property
    def total_size(self):
        return self._size

    def clear(self):
        self._size = 0
        self.od.clear()

    def popitem(self, last=True):
        k, v = self.od.popitem(last=last)
        self._size -= self._get_value_size(v)

        return k, v

    def pop(self, key):
        v = self.od.pop(key)
        self._size -= self._get_value_size(v)

        return v

    def _adjust_size(self, key, value):
        if key in self.od:
            self._size -= self._get_value_size(self.od[key])

        self._size += self._get_value_size(value)

    @abc.abstractmethod
    def _get_value_size(self, value):
        raise NotImplementedError


class MemCacheLengthUnit(MemCacheUnit):
    """基于长度限制的内存缓存单元"""
    def __init__(self, size_limit=0):
        super().__init__(size_limit=size_limit)

    def _get_value_size(self, value):
        return 1


class MemCacheSizeofUnit(MemCacheUnit):
    """基于内存大小（sizeof）限制的内存缓存单元"""
    def __init__(self, size_limit=0):
        super().__init__(size_limit=size_limit)

    def _get_value_size(self, value):
        return sys.getsizeof(value)


class MemCache:
    """内存缓存"""

    def __init__(self, mem_cache_size_limit=None, limit_type="length"):
        """
        参数
        ----------
        mem_cache_size_limit:
            缓存最大大小。
        limit_type:
            'length' 或 'sizeof'；'length'（调用 len()），'sizeof'（调用 sys.getsizeof()）。
        """

        size_limit = C.mem_cache_size_limit if mem_cache_size_limit is None else mem_cache_size_limit
        limit_type = C.mem_cache_limit_type if limit_type is None else limit_type

        if limit_type == "length":
            klass = MemCacheLengthUnit
        elif limit_type == "sizeof":
            klass = MemCacheSizeofUnit
        else:
            raise ValueError(f"limit_type 必须是 'length' 或 'sizeof'，但你提供的是 {limit_type}")

        self.__calendar_mem_cache = klass(size_limit)
        self.__instrument_mem_cache = klass(size_limit)
        self.__feature_mem_cache = klass(size_limit)

    def __getitem__(self, key):
        if key == "c":
            return self.__calendar_mem_cache
        elif key == "i":
            return self.__instrument_mem_cache
        elif key == "f":
            return self.__feature_mem_cache
        else:
            raise KeyError("未知的内存缓存单元")

    def clear(self):
        self.__calendar_mem_cache.clear()
        self.__instrument_mem_cache.clear()
        self.__feature_mem_cache.clear()


class MemCacheExpire:
    """内存缓存过期管理"""
    CACHE_EXPIRE = C.mem_cache_expire

    @staticmethod
    def set_cache(mem_cache, key, value):
        """设置缓存

        :param mem_cache: MemCache 属性（'c'/'i'/'f'）。
        :param key: 缓存键。
        :param value: 缓存值。
        """
        mem_cache[key] = value, time.time()

    @staticmethod
    def get_cache(mem_cache, key):
        """获取内存缓存

        :param mem_cache: MemCache 属性（'c'/'i'/'f'）。
        :param key: 缓存键。
        :return: 缓存值；如果缓存不存在，则返回 None。
        """
        value = None
        expire = False
        if key in mem_cache:
            value, latest_time = mem_cache[key]
            expire = (time.time() - latest_time) > MemCacheExpire.CACHE_EXPIRE
        return value, expire


class CacheUtils:
    """缓存工具类"""
    LOCK_ID = "QLIB"

    @staticmethod
    def organize_meta_file():
        pass

    @staticmethod
    def reset_lock():
        """重置 Redis 锁"""
        r = get_redis_connection()
        redis_lock.reset_all(r)

    @staticmethod
    def visit(cache_path: Union[str, Path]):
        """记录缓存访问信息"""
        # FIXME: 因为读取缓存时取消了读锁，多个进程在这里可能会出现读写异常
        try:
            cache_path = Path(cache_path)
            meta_path = cache_path.with_suffix(".meta")
            with meta_path.open("rb") as f:
                d = pickle.load(f)
            with meta_path.open("wb") as f:
                try:
                    d["meta"]["last_visit"] = str(time.time())
                    d["meta"]["visits"] = d["meta"]["visits"] + 1
                except KeyError as key_e:
                    raise KeyError("未知的 meta 关键字") from key_e
                pickle.dump(d, f, protocol=C.dump_protocol_version)
        except Exception as e:
            get_module_logger("CacheUtils").warning(f"访问 {cache_path} 缓存时出错: {e}")

    @staticmethod
    def acquire(lock, lock_name):
        """获取锁"""
        try:
            lock.acquire()
        except redis_lock.AlreadyAcquired as lock_acquired:
            raise QlibCacheException(
                f"""看起来 redis 锁的键 (lock:{repr(lock_name)[1:-1]}-wlock) 已经存在于你的 redis 数据库中。
                    你可以使用以下命令清除你的 redis 键并重新运行你的命令：
                    $ redis-cli
                    > select {C.redis_task_db}
                    > del "lock:{repr(lock_name)[1:-1]}-wlock"
                    > quit
                    如果问题仍未解决，请使用 "keys *" 查找是否存在多个键。如果是，请尝试使用 "flushall" 清除所有键。
                """
            ) from lock_acquired

    @staticmethod
    @contextlib.contextmanager
    def reader_lock(redis_t, lock_name: str):
        """读锁上下文管理器"""
        current_cache_rlock = redis_lock.Lock(redis_t, f"{lock_name}-rlock")
        current_cache_wlock = redis_lock.Lock(redis_t, f"{lock_name}-wlock")
        lock_reader = f"{lock_name}-reader"
        # 确保只有一个读进程进入
        current_cache_rlock.acquire(timeout=60)
        try:
            current_cache_readers = redis_t.get(lock_reader)
            if current_cache_readers is None or int(current_cache_readers) == 0:
                CacheUtils.acquire(current_cache_wlock, lock_name)
            redis_t.incr(lock_reader)
        finally:
            current_cache_rlock.release()
        try:
            yield
        finally:
            # 确保只有一个读进程离开
            current_cache_rlock.acquire(timeout=60)
            try:
                redis_t.decr(lock_reader)
                if int(redis_t.get(lock_reader)) == 0:
                    redis_t.delete(lock_reader)
                    current_cache_wlock.reset()
            finally:
                current_cache_rlock.release()

    @staticmethod
    @contextlib.contextmanager
    def writer_lock(redis_t, lock_name):
        """写锁上下文管理器"""
        current_cache_wlock = redis_lock.Lock(redis_t, f"{lock_name}-wlock", id=CacheUtils.LOCK_ID)
        CacheUtils.acquire(current_cache_wlock, lock_name)
        try:
            yield
        finally:
            current_cache_wlock.release()


class BaseProviderCache:
    """提供程序缓存基类"""

    def __init__(self, provider):
        self.provider = provider
        self.logger = get_module_logger(self.__class__.__name__)

    def __getattr__(self, attr):
        return getattr(self.provider, attr)

    @staticmethod
    def check_cache_exists(cache_path: Union[str, Path], suffix_list: Iterable = (".index", ".meta")) -> bool:
        """检查缓存是否存在"""
        cache_path = Path(cache_path)
        for p in [cache_path] + [cache_path.with_suffix(_s) for _s in suffix_list]:
            if not p.exists():
                return False
        return True

    @staticmethod
    def clear_cache(cache_path: Union[str, Path]):
        """清除缓存"""
        for p in [
            cache_path,
            cache_path.with_suffix(".meta"),
            cache_path.with_suffix(".index"),
        ]:
            if p.exists():
                p.unlink()

    @staticmethod
    def get_cache_dir(dir_name: str, freq: str = None) -> Path:
        """获取缓存目录"""
        cache_dir = Path(C.dpm.get_data_uri(freq)).joinpath(dir_name)
        cache_dir.mkdir(parents=True, exist_ok=True)
        return cache_dir


class ExpressionCache(BaseProviderCache):
    """表达式缓存机制基类。

    此类用于包装具有自定义表达式缓存机制的表达式提供程序。

    .. note:: 覆盖 `_uri` 和 `_expression` 方法以创建自己的表达式缓存机制。
    """

    def expression(self, instrument, field, start_time, end_time, freq):
        """获取表达式数据。

        .. note:: 与表达式提供程序中的 `expression` 方法具有相同的接口
        """
        try:
            return self._expression(instrument, field, start_time, end_time, freq)
        except NotImplementedError:
            return self.provider.expression(instrument, field, start_time, end_time, freq)

    def _uri(self, instrument, field, start_time, end_time, freq):
        """获取表达式缓存文件 URI。

        覆盖此方法以定义如何根据用户自己的缓存机制获取表达式缓存文件 URI。
        """
        raise NotImplementedError("实现此函数以匹配您自己的缓存机制")

    def _expression(self, instrument, field, start_time, end_time, freq):
        """使用缓存获取表达式数据。

        覆盖此方法以定义如何根据用户自己的缓存机制获取表达式数据。
        """
        raise NotImplementedError("如果要使用表达式缓存，请实现此方法")

    def update(self, cache_uri: Union[str, Path], freq: str = "day"):
        """将表达式缓存更新到最新的日历。

        覆盖此方法以定义如何根据用户自己的缓存机制更新表达式缓存。

        参数
        ----------
        cache_uri : str or Path
            表达式缓存文件的完整 URI（包括目录路径）。
        freq : str

        返回
        -------
        int
            0（更新成功）/ 1（无需更新）/ 2（更新失败）。
        """
        raise NotImplementedError("如果要使表达式缓存保持最新，请实现此方法")


class DatasetCache(BaseProviderCache):
    """数据集缓存机制基类。

    此类用于包装具有自定义数据集缓存机制的数据集提供程序。

    .. note:: 覆盖 `_uri` 和 `_dataset` 方法以创建自己的数据集缓存机制。
    """

    HDF_KEY = "df"

    def dataset(
        self, instruments, fields, start_time=None, end_time=None, freq="day", disk_cache=1, inst_processors=[]
    ):
        """获取特征数据集。

        .. note:: 与数据集提供程序中的 `dataset` 方法具有相同的接口

        .. note:: 服务器使用 redis_lock 确保不会触发读写冲突，但未考虑客户端读取器。
        """
        if disk_cache == 0:
            # 跳过缓存
            return self.provider.dataset(
                instruments, fields, start_time, end_time, freq, inst_processors=inst_processors
            )
        else:
            # 使用并替换缓存
            try:
                return self._dataset(
                    instruments, fields, start_time, end_time, freq, disk_cache, inst_processors=inst_processors
                )
            except NotImplementedError:
                return self.provider.dataset(
                    instruments, fields, start_time, end_time, freq, inst_processors=inst_processors
                )

    def _uri(self, instruments, fields, start_time, end_time, freq, **kwargs):
        """获取数据集缓存文件 URI。

        覆盖此方法以定义如何根据用户自己的缓存机制获取数据集缓存文件 URI。
        """
        raise NotImplementedError("实现此函数以匹配您自己的缓存机制")

    def _dataset(
        self, instruments, fields, start_time=None, end_time=None, freq="day", disk_cache=1, inst_processors=[]
    ):
        """使用缓存获取特征数据集。

        覆盖此方法以定义如何根据用户自己的缓存机制获取特征数据集。
        """
        raise NotImplementedError("如果要使用数据集特征缓存，请实现此方法")

    def _dataset_uri(
        self, instruments, fields, start_time=None, end_time=None, freq="day", disk_cache=1, inst_processors=[]
    ):
        """使用缓存获取特征数据集的 URI。
        特别地：
            disk_cache=1 表示使用数据集缓存并返回缓存文件的 URI。
            disk_cache=0 表示客户端知道表达式缓存的路径，
                         服务器检查缓存是否存在（如果不存在，则生成它），然后客户端自行加载数据。
        覆盖此方法以定义如何根据用户自己的缓存机制获取特征数据集 URI。
        """
        raise NotImplementedError(
            "如果要将数据集特征缓存用作客户端的缓存文件，请实现此方法"
        )

    def update(self, cache_uri: Union[str, Path], freq: str = "day"):
        """将数据集缓存更新到最新的日历。

        覆盖此方法以定义如何根据用户自己的缓存机制更新数据集缓存。

        参数
        ----------
        cache_uri : str or Path
            数据集缓存文件的完整 URI（包括目录路径）。
        freq : str

        返回
        -------
        int
            0（更新成功）/ 1（无需更新）/ 2（更新失败）
        """
        raise NotImplementedError("如果要使表达式缓存保持最新，请实现此方法")

    @staticmethod
    def cache_to_origin_data(data, fields):
        """将缓存数据转换为原始数据

        :param data: pd.DataFrame，缓存数据。
        :param fields: 特征字段。
        :return: pd.DataFrame.
        """
        not_space_fields = remove_fields_space(fields)
        data = data.loc[:, not_space_fields]
        # 设置特征字段
        data.columns = [str(i) for i in fields]
        return data

    @staticmethod
    def normalize_uri_args(instruments, fields, freq):
        """规范化 URI 参数"""
        instruments = normalize_cache_instruments(instruments)
        fields = normalize_cache_fields(fields)
        freq = freq.lower()

        return instruments, fields, freq


class DiskExpressionCache(ExpressionCache):
    """为服务器准备的磁盘表达式缓存机制。"""

    def __init__(self, provider, **kwargs):
        super(DiskExpressionCache, self).__init__(provider)
        self.r = get_redis_connection()
        # remote==True 表示客户端正在使用此模块，将不允许写入行为。
        self.remote = kwargs.get("remote", False)

    def get_cache_dir(self, freq: str = None) -> Path:
        return super(DiskExpressionCache, self).get_cache_dir(C.features_cache_dir_name, freq)

    def _uri(self, instrument, field, start_time, end_time, freq):
        field = remove_fields_space(field)
        instrument = str(instrument).lower()
        return hash_args(instrument, field, freq)

    def _expression(self, instrument, field, start_time=None, end_time=None, freq="day"):
        _cache_uri = self._uri(instrument=instrument, field=field, start_time=None, end_time=None, freq=freq)
        _instrument_dir = self.get_cache_dir(freq).joinpath(instrument.lower())
        cache_path = _instrument_dir.joinpath(_cache_uri)
        # 获取日历
        from .data import Cal  # pylint: disable=C0415

        _calendar = Cal.calendar(freq=freq)

        _, _, start_index, end_index = Cal.locate_index(start_time, end_time, freq, future=False)

        if self.check_cache_exists(cache_path, suffix_list=[".meta"]):
            """
            在大多数情况下，我们不需要读锁。
            因为与读取数据相比，更新数据是小概率事件。
            """
            # FIXME: 删除读锁可能会导致冲突。
            # with CacheUtils.reader_lock(self.r, 'expression-%s' % _cache_uri):

            # 修改表达式缓存元文件
            try:
                # FIXME: 多个读取器可能会导致访问次数错误
                if not self.remote:
                    CacheUtils.visit(cache_path)
                series = read_bin(cache_path, start_index, end_index)
                return series
            except Exception:
                series = None
                self.logger.error("读取 %s 文件时出错 : %s" % (cache_path, traceback.format_exc()))
            return series
        else:
            # 规范化字段
            field = remove_fields_space(field)
            # 缓存不可用，生成缓存
            _instrument_dir.mkdir(parents=True, exist_ok=True)
            if not isinstance(eval(parse_field(field)), Feature):
                # 当表达式不是原始特征时
                # 如果特征不是 Feature 实例，则生成表达式缓存
                series = self.provider.expression(instrument, field, _calendar[0], _calendar[-1], freq)
                if not series.empty:
                    # 此表达式为空，我们不为其生成任何缓存。
                    with CacheUtils.writer_lock(self.r, f"{str(C.dpm.get_data_uri(freq))}:expression-{_cache_uri}"):
                        self.gen_expression_cache(
                            expression_data=series,
                            cache_path=cache_path,
                            instrument=instrument,
                            field=field,
                            freq=freq,
                            last_update=str(_calendar[-1]),
                        )
                    return series.loc[start_index:end_index]
                else:
                    return series
            else:
                # 如果表达式是原始特征（例如 $close, $open）
                return self.provider.expression(instrument, field, start_time, end_time, freq)

    def gen_expression_cache(self, expression_data, cache_path, instrument, field, freq, last_update):
        """使用二进制文件保存类似特征的数据。"""
        # 确保在目录被删除时缓存仍能正常运行
        meta = {
            "info": {"instrument": instrument, "field": field, "freq": freq, "last_update": last_update},
            "meta": {"last_visit": time.time(), "visits": 1},
        }
        self.logger.debug(f"正在生成表达式缓存: {meta}")
        self.clear_cache(cache_path)
        meta_path = cache_path.with_suffix(".meta")

        with meta_path.open("wb") as f:
            pickle.dump(meta, f, protocol=C.dump_protocol_version)
        meta_path.chmod(stat.S_IRWXU | stat.S_IRGRP | stat.S_IROTH)
        df = expression_data.to_frame()

        r = np.hstack([df.index[0], expression_data]).astype("<f")
        r.tofile(str(cache_path))

    def update(self, sid, cache_uri, freq: str = "day"):
        cp_cache_uri = self.get_cache_dir(freq).joinpath(sid).joinpath(cache_uri)
        meta_path = cp_cache_uri.with_suffix(".meta")
        if not self.check_cache_exists(cp_cache_uri, suffix_list=[".meta"]):
            self.logger.info(f"缓存 {cp_cache_uri} 已损坏，将被删除")
            self.clear_cache(cp_cache_uri)
            return 2

        with CacheUtils.writer_lock(self.r, f"{str(C.dpm.get_data_uri())}:expression-{cache_uri}"):
            with meta_path.open("rb") as f:
                d = pickle.load(f)
            instrument = d["info"]["instrument"]
            field = d["info"]["field"]
            freq = d["info"]["freq"]
            last_update_time = d["info"]["last_update"]

            # 获取最新的日历
            from .data import Cal, ExpressionD  # pylint: disable=C0415

            whole_calendar = Cal.calendar(start_time=None, end_time=None, freq=freq)
            # 自上次更新以来的日历。
            new_calendar = Cal.calendar(start_time=last_update_time, end_time=None, freq=freq)

            # 获取追加数据
            if len(new_calendar) <= 1:
                # 包括上次更新的日历，我们只得到 1 个项目。
                # 不需要未来更新。
                return 1
            else:
                # 获取删除历史数据后所需的数据。
                # 新数据的开始索引
                current_index = len(whole_calendar) - len(new_calendar) + 1

                # 现有数据长度
                size_bytes = os.path.getsize(cp_cache_uri)
                ele_size = np.dtype("<f").itemsize
                assert size_bytes % ele_size == 0
                ele_n = size_bytes // ele_size - 1

                expr = ExpressionD.get_expression_instance(field)
                lft_etd, rght_etd = expr.get_extended_window_size()
                # 表达式使用了 rght_etd 天后的未来数据。
                # 因此应删除最后的 rght_etd 数据。
                # 最多可以删除 `ele_n` 个周期的数据
                remove_n = min(rght_etd, ele_n)
                assert new_calendar[1] == whole_calendar[current_index]
                data = self.provider.expression(
                    instrument, field, whole_calendar[current_index - remove_n], new_calendar[-1], freq
                )
                with open(cp_cache_uri, "ab") as f:
                    data = np.array(data).astype("<f")
                    # 删除最后的位
                    f.truncate(size_bytes - ele_size * remove_n)
                    f.write(data)
                # 更新元文件
                d["info"]["last_update"] = str(new_calendar[-1])
                with meta_path.open("wb") as f:
                    pickle.dump(d, f, protocol=C.dump_protocol_version)
        return 0


class DiskDatasetCache(DatasetCache):
    """为服务器准备的磁盘数据集缓存机制。"""

    def __init__(self, provider, **kwargs):
        super(DiskDatasetCache, self).__init__(provider)
        self.r = get_redis_connection()
        self.remote = kwargs.get("remote", False)

    @staticmethod
    def _uri(instruments, fields, start_time, end_time, freq, disk_cache=1, inst_processors=[], **kwargs):
        return hash_args(*DatasetCache.normalize_uri_args(instruments, fields, freq), disk_cache, inst_processors)

    def get_cache_dir(self, freq: str = None) -> Path:
        return super(DiskDatasetCache, self).get_cache_dir(C.dataset_cache_dir_name, freq)

    @classmethod
    def read_data_from_cache(cls, cache_path: Union[str, Path], start_time, end_time, fields):
        """从缓存中读取数据

        此函数可以从磁盘缓存数据集中读取数据

        :param cache_path:
        :param start_time:
        :param end_time:
        :param fields: 数据集缓存的字段顺序是排序的。因此重新排列列以使其一致。
        :return:
        """

        im = DiskDatasetCache.IndexManager(cache_path)
        index_data = im.get_index(start_time, end_time)
        if index_data.shape[0] > 0:
            start, stop = (
                index_data["start"].iloc[0].item(),
                index_data["end"].iloc[-1].item(),
            )
        else:
            start = stop = 0

        with pd.HDFStore(cache_path, mode="r") as store:
            if "/{}".format(im.KEY) in store.keys():
                df = store.select(key=im.KEY, start=start, stop=stop)
                df = df.swaplevel("datetime", "instrument").sort_index()
                # 读取缓存并需要将非空格字段替换为字段
                df = cls.cache_to_origin_data(df, fields)

            else:
                df = pd.DataFrame(columns=fields)
        return df

    def _dataset(
        self, instruments, fields, start_time=None, end_time=None, freq="day", disk_cache=0, inst_processors=[]
    ):
        if disk_cache == 0:
            # 在这种情况下，数据集缓存已配置但不会被使用。
            return self.provider.dataset(
                instruments, fields, start_time, end_time, freq, inst_processors=inst_processors
            )
        # FIXME: 重采样后的缓存在再次读取并使用 end_time 截取时，会导致数据日期不完整
        if inst_processors:
            raise ValueError(
                f"{self.__class__.__name__} 不支持 inst_processor。 "
                f"请使用 `D.features(disk_cache=0)` 或 `qlib.init(dataset_cache=None)`"
            )
        _cache_uri = self._uri(
            instruments=instruments,
            fields=fields,
            start_time=None,
            end_time=None,
            freq=freq,
            disk_cache=disk_cache,
            inst_processors=inst_processors,
        )

        cache_path = self.get_cache_dir(freq).joinpath(_cache_uri)

        features = pd.DataFrame()
        gen_flag = False

        if self.check_cache_exists(cache_path):
            if disk_cache == 1:
                # 使用缓存
                with CacheUtils.reader_lock(self.r, f"{str(C.dpm.get_data_uri(freq))}:dataset-{_cache_uri}"):
                    CacheUtils.visit(cache_path)
                    features = self.read_data_from_cache(cache_path, start_time, end_time, fields)
            elif disk_cache == 2:
                gen_flag = True
        else:
            gen_flag = True

        if gen_flag:
            # 缓存不可用，生成缓存
            with CacheUtils.writer_lock(self.r, f"{str(C.dpm.get_data_uri(freq))}:dataset-{_cache_uri}"):
                features = self.gen_dataset_cache(
                    cache_path=cache_path,
                    instruments=instruments,
                    fields=fields,
                    freq=freq,
                    inst_processors=inst_processors,
                )
            if not features.empty:
                features = features.sort_index().loc(axis=0)[:, start_time:end_time]
        return features

    def _dataset_uri(
        self, instruments, fields, start_time=None, end_time=None, freq="day", disk_cache=0, inst_processors=[]
    ):
        if disk_cache == 0:
            # 在这种情况下，服务器只检查表达式缓存。
            # 客户端将自行加载缓存数据。
            from .data import LocalDatasetProvider  # pylint: disable=C0415

            LocalDatasetProvider.multi_cache_walker(instruments, fields, start_time, end_time, freq)
            return ""
        # FIXME: 重采样后的缓存在再次读取并使用 end_time 截取时，会导致数据日期不完整
        if inst_processors:
            raise ValueError(
                f"{self.__class__.__name__} 不支持 inst_processor。 "
                f"请使用 `D.features(disk_cache=0)` 或 `qlib.init(dataset_cache=None)`"
            )
        _cache_uri = self._uri(
            instruments=instruments,
            fields=fields,
            start_time=None,
            end_time=None,
            freq=freq,
            disk_cache=disk_cache,
            inst_processors=inst_processors,
        )
        cache_path = self.get_cache_dir(freq).joinpath(_cache_uri)

        if self.check_cache_exists(cache_path):
            self.logger.debug(f"缓存数据集已存在于 {cache_path}。直接返回 URI")
            with CacheUtils.reader_lock(self.r, f"{str(C.dpm.get_data_uri(freq))}:dataset-{_cache_uri}"):
                CacheUtils.visit(cache_path)
            return _cache_uri
        else:
            # 缓存不可用，生成缓存
            with CacheUtils.writer_lock(self.r, f"{str(C.dpm.get_data_uri(freq))}:dataset-{_cache_uri}"):
                self.gen_dataset_cache(
                    cache_path=cache_path,
                    instruments=instruments,
                    fields=fields,
                    freq=freq,
                    inst_processors=inst_processors,
                )
            return _cache_uri

    class IndexManager:
        """
        此类不考虑锁。请在代码外部考虑锁。
        此类是磁盘数据的代理。
        """

        KEY = "df"

        def __init__(self, cache_path: Union[str, Path]):
            self.index_path = cache_path.with_suffix(".index")
            self._data = None
            self.logger = get_module_logger(self.__class__.__name__)

        def get_index(self, start_time=None, end_time=None):
            # TODO: 从磁盘快速读取索引。
            if self._data is None:
                self.sync_from_disk()
            return self._data.loc[start_time:end_time].copy()

        def sync_to_disk(self):
            if self._data is None:
                raise ValueError("没有数据可同步到磁盘。")
            self._data.sort_index(inplace=True)
            self._data.to_hdf(self.index_path, key=self.KEY, mode="w", format="table")
            # 索引应为所有用户可读
            self.index_path.chmod(stat.S_IRWXU | stat.S_IRGRP | stat.S_IROTH)

        def sync_from_disk(self):
            # 如果直接从磁盘 read_hdf，文件不会直接关闭
            with pd.HDFStore(self.index_path, mode="r") as store:
                if "/{}".format(self.KEY) in store.keys():
                    self._data = pd.read_hdf(store, key=self.KEY)
                else:
                    self._data = pd.DataFrame()

        def update(self, data, sync=True):
            self._data = data.astype(np.int32).copy()
            if sync:
                self.sync_to_disk()

        def append_index(self, data, to_disk=True):
            data = data.astype(np.int32).copy()
            data.sort_index(inplace=True)
            self._data = pd.concat([self._data, data])
            if to_disk:
                with pd.HDFStore(self.index_path) as store:
                    store.append(self.KEY, data, append=True)

        @staticmethod
        def build_index_from_data(data, start_index=0):
            if data.empty:
                return pd.DataFrame()
            line_data = data.groupby("datetime", group_keys=False).size()
            line_data.sort_index(inplace=True)
            index_end = line_data.cumsum()
            index_start = index_end.shift(1, fill_value=0)

            index_data = pd.DataFrame()
            index_data["start"] = index_start
            index_data["end"] = index_end
            index_data += start_index
            return index_data

    def gen_dataset_cache(self, cache_path: Union[str, Path], instruments, fields, freq, inst_processors=[]):
        """生成数据集缓存

        .. note:: 此函数不考虑缓存读写锁。请在此函数外部获取锁

        缓存格式包含 3 部分（后跟典型文件名）。

        - 索引 : cache/d41366901e25de3ec47297f12e2ba11d.index

            - 文件内容可能采用以下格式（pandas.Series）

                .. code-block:: python

                                        start end
                    1999-11-10 00:00:00     0   1
                    1999-11-11 00:00:00     1   2
                    1999-11-12 00:00:00     2   3
                    ...

                .. note:: start 是闭区间，end 是开区间！！！！

            - 每行包含两个元素 <start_index, end_index>，并以时间戳作为其索引。
            - 它指示 `timestamp` 的数据的 `start_index`（包含）和 `end_index`（不包含）。

        - 元数据: cache/d41366901e25de3ec47297f12e2ba11d.meta

        - 数据     : cache/d41366901e25de3ec47297f12e2ba11d

            - 这是一个按日期时间排序的 hdf 文件

        :param cache_path:  存储缓存的路径。
        :param instruments:  存储缓存的金融工具。
        :param fields:  存储缓存的字段。
        :param freq:  存储缓存的频率。
        :param inst_processors:  金融工具处理器。

        :return type pd.DataFrame; 返回的 DataFrame 的字段与函数的参数一致。
        """
        # 获取日历
        from .data import Cal  # pylint: disable=C0415

        cache_path = Path(cache_path)
        _calendar = Cal.calendar(freq=freq)
        self.logger.debug(f"正在生成数据集缓存 {cache_path}")
        # 确保在目录被删除时缓存仍能正常运行
        self.clear_cache(cache_path)

        features = self.provider.dataset(
            instruments, fields, _calendar[0], _calendar[-1], freq, inst_processors=inst_processors
        )

        if features.empty:
            return features

        # 交换索引级别并排序
        features = features.swaplevel("instrument", "datetime").sort_index()

        # 写入缓存数据
        with pd.HDFStore(str(cache_path.with_suffix(".data"))) as store:
            cache_to_orig_map = dict(zip(remove_fields_space(features.columns), features.columns))
            orig_to_cache_map = dict(zip(features.columns, remove_fields_space(features.columns)))
            cache_features = features[list(cache_to_orig_map.values())].rename(columns=orig_to_cache_map)
            # 缓存列
            cache_columns = sorted(cache_features.columns)
            cache_features = cache_features.loc[:, cache_columns]
            cache_features = cache_features.loc[:, ~cache_features.columns.duplicated()]
            store.append(DatasetCache.HDF_KEY, cache_features, append=False)
        # 写入元文件
        meta = {
            "info": {
                "instruments": instruments,
                "fields": list(cache_features.columns),
                "freq": freq,
                "last_update": str(_calendar[-1]),  # 存储缓存的最后更新
                "inst_processors": inst_processors,  # 存储缓存的金融工具处理器
            },
            "meta": {"last_visit": time.time(), "visits": 1},
        }
        with cache_path.with_suffix(".meta").open("wb") as f:
            pickle.dump(meta, f, protocol=C.dump_protocol_version)
        cache_path.with_suffix(".meta").chmod(stat.S_IRWXU | stat.S_IRGRP | stat.S_IROTH)
        # 写入索引文件
        im = DiskDatasetCache.IndexManager(cache_path)
        index_data = im.build_index_from_data(features)
        im.update(index_data)

        # 缓存生成后重命名文件
        # 这在 windows 上效果不佳，但我们的服务器暂时不会使用 windows
        cache_path.with_suffix(".data").rename(cache_path)
        # 缓存特征的字段将转换为原始字段
        return features.swaplevel("datetime", "instrument")

    def update(self, cache_uri, freq: str = "day"):
        cp_cache_uri = self.get_cache_dir(freq).joinpath(cache_uri)
        meta_path = cp_cache_uri.with_suffix(".meta")
        if not self.check_cache_exists(cp_cache_uri):
            self.logger.info(f"缓存 {cp_cache_uri} 已损坏，将被删除")
            self.clear_cache(cp_cache_uri)
            return 2

        im = DiskDatasetCache.IndexManager(cp_cache_uri)
        with CacheUtils.writer_lock(self.r, f"{str(C.dpm.get_data_uri())}:dataset-{cache_uri}"):
            with meta_path.open("rb") as f:
                d = pickle.load(f)
            instruments = d["info"]["instruments"]
            fields = d["info"]["fields"]
            freq = d["info"]["freq"]
            last_update_time = d["info"]["last_update"]
            inst_processors = d["info"].get("inst_processors", [])
            index_data = im.get_index()

            self.logger.debug("正在更新数据集: {}".format(d))
            from .data import Inst  # pylint: disable=C0415

            if Inst.get_inst_type(instruments) == Inst.DICT:
                self.logger.info(f"文件 {cache_uri} 包含字典缓存。跳过更新")
                return 1

            # 获取最新的日历
            from .data import Cal  # pylint: disable=C0415

            whole_calendar = Cal.calendar(start_time=None, end_time=None, freq=freq)
            # 自上次更新以来的日历
            new_calendar = Cal.calendar(start_time=last_update_time, end_time=None, freq=freq)

            # 获取追加数据
            if len(new_calendar) <= 1:
                # 包括上次更新的日历，我们只得到 1 个项目。
                # 不需要未来更新。
                return 1
            else:
                # 获取删除历史数据后所需的数据。
                # 新数据的开始索引
                current_index = len(whole_calendar) - len(new_calendar) + 1

                # 避免递归导入
                from .data import ExpressionD  # pylint: disable=C0415

                # 现有数据长度
                lft_etd = rght_etd = 0
                for field in fields:
                    expr = ExpressionD.get_expression_instance(field)
                    l, r = expr.get_extended_window_size()
                    lft_etd = max(lft_etd, l)
                    rght_etd = max(rght_etd, r)
                # 删除应更新的周期。
                if index_data.empty:
                    # 我们没有任何此数据集的数据。无需删除
                    rm_n_period = rm_lines = 0
                else:
                    rm_n_period = min(rght_etd, index_data.shape[0])
                    rm_lines = (
                        (index_data["end"] - index_data["start"])
                        .loc[whole_calendar[current_index - rm_n_period] :]
                        .sum()
                        .item()
                    )

                data = self.provider.dataset(
                    instruments,
                    fields,
                    whole_calendar[current_index - rm_n_period],
                    new_calendar[-1],
                    freq,
                    inst_processors=inst_processors,
                )

                if not data.empty:
                    data.reset_index(inplace=True)
                    data.set_index(["datetime", "instrument"], inplace=True)
                    data.sort_index(inplace=True)
                else:
                    return 0  # 没有数据可更新缓存

                store = pd.HDFStore(cp_cache_uri)
                # FIXME:
                # 因为特征缓存存储为 .bin 文件。
                # 所以从特征读取的序列都是 float32。
                # 然而，第一个数据集缓存是基于原始数据计算的。
                # 所以数据类型可能是 float64。
                # 不同的数据类型会导致追加数据失败
                if "/{}".format(DatasetCache.HDF_KEY) in store.keys():
                    schema = store.select(DatasetCache.HDF_KEY, start=0, stop=0)
                    for col, dtype in schema.dtypes.items():
                        data[col] = data[col].astype(dtype)
                if rm_lines > 0:
                    store.remove(key=im.KEY, start=-rm_lines)
                store.append(DatasetCache.HDF_KEY, data)
                store.close()

                # 更新索引文件
                new_index_data = im.build_index_from_data(
                    data.loc(axis=0)[whole_calendar[current_index] :, :],
                    start_index=0 if index_data.empty else index_data["end"].iloc[-1],
                )
                im.append_index(new_index_data)

                # 更新元文件
                d["info"]["last_update"] = str(new_calendar[-1])
                with meta_path.open("wb") as f:
                    pickle.dump(d, f, protocol=C.dump_protocol_version)
                return 0


class SimpleDatasetCache(DatasetCache):
    """可本地或客户端使用的简单数据集缓存。"""

    def __init__(self, provider):
        super(SimpleDatasetCache, self).__init__(provider)
        try:
            self.local_cache_path: Path = Path(C["local_cache_path"]).expanduser().resolve()
        except (KeyError, TypeError):
            self.logger.error("如果要使用此缓存机制，请在配置中分配一个 local_cache_path")
            raise
        self.logger.info(
            f"数据集缓存目录: {self.local_cache_path}, "
            f"通过配置中的 local_cache_path 修改缓存目录"
        )

    def _uri(self, instruments, fields, start_time, end_time, freq, disk_cache=1, inst_processors=[], **kwargs):
        instruments, fields, freq = self.normalize_uri_args(instruments, fields, freq)
        return hash_args(
            instruments, fields, start_time, end_time, freq, disk_cache, str(self.local_cache_path), inst_processors
        )

    def _dataset(
        self, instruments, fields, start_time=None, end_time=None, freq="day", disk_cache=1, inst_processors=[]
    ):
        if disk_cache == 0:
            # 在这种情况下，数据集缓存已配置但不会被使用。
            return self.provider.dataset(instruments, fields, start_time, end_time, freq)
        self.local_cache_path.mkdir(exist_ok=True, parents=True)
        cache_file = self.local_cache_path.joinpath(
            self._uri(
                instruments, fields, start_time, end_time, freq, disk_cache=disk_cache, inst_processors=inst_processors
            )
        )
        gen_flag = False

        if cache_file.exists():
            if disk_cache == 1:
                # 使用缓存
                df = pd.read_pickle(cache_file)
                return self.cache_to_origin_data(df, fields)
            elif disk_cache == 2:
                # 替换缓存
                gen_flag = True
        else:
            gen_flag = True

        if gen_flag:
            data = self.provider.dataset(
                instruments, normalize_cache_fields(fields), start_time, end_time, freq, inst_processors=inst_processors
            )
            data.to_pickle(cache_file)
            return self.cache_to_origin_data(data, fields)


class DatasetURICache(DatasetCache):
    """为服务器准备的数据集 URI 缓存机制。"""

    def _uri(self, instruments, fields, start_time, end_time, freq, disk_cache=1, inst_processors=[], **kwargs):
        return hash_args(*self.normalize_uri_args(instruments, fields, freq), disk_cache, inst_processors)

    def dataset(
        self, instruments, fields, start_time=None, end_time=None, freq="day", disk_cache=0, inst_processors=[]
    ):
        if "local" in C.dataset_provider.lower():
            # 使用 LocalDatasetProvider
            return self.provider.dataset(
                instruments, fields, start_time, end_time, freq, inst_processors=inst_processors
            )

        if disk_cache == 0:
            # 不使用数据集缓存，直接从远程表达式缓存加载数据
            return self.provider.dataset(
                instruments,
                fields,
                start_time,
                end_time,
                freq,
                disk_cache,
                return_uri=False,
                inst_processors=inst_processors,
            )
        # FIXME: 重采样后的缓存在再次读取并使用 end_time 截取时，会导致数据日期不完整
        if inst_processors:
            raise ValueError(
                f"{self.__class__.__name__} 不支持 inst_processor。 "
                f"请使用 `D.features(disk_cache=0)` 或 `qlib.init(dataset_cache=None)`"
            )
        # 使用 ClientDatasetProvider
        feature_uri = self._uri(
            instruments, fields, None, None, freq, disk_cache=disk_cache, inst_processors=inst_processors
        )
        value, expire = MemCacheExpire.get_cache(H["f"], feature_uri)
        mnt_feature_uri = C.dpm.get_data_uri(freq).joinpath(C.dataset_cache_dir_name).joinpath(feature_uri)
        if value is None or expire or not mnt_feature_uri.exists():
            df, uri = self.provider.dataset(
                instruments,
                fields,
                start_time,
                end_time,
                freq,
                disk_cache,
                return_uri=True,
                inst_processors=inst_processors,
            )
            # 缓存 URI
            MemCacheExpire.set_cache(H["f"], uri, uri)
            # 缓存 DataFrame
            # HZ['f'][uri] = df.copy()
            get_module_logger("cache").debug(f"从 {C.dataset_provider} 获取特征")
        else:
            df = DiskDatasetCache.read_data_from_cache(mnt_feature_uri, start_time, end_time, fields)
            get_module_logger("cache").debug("从 URI 缓存获取特征")

        return df


class CalendarCache(BaseProviderCache):
    """日历缓存基类"""
    pass


class MemoryCalendarCache(CalendarCache):
    """内存日历缓存"""
    def calendar(self, start_time=None, end_time=None, freq="day", future=False):
        uri = self._uri(start_time, end_time, freq, future)
        result, expire = MemCacheExpire.get_cache(H["c"], uri)
        if result is None or expire:
            result = self.provider.calendar(start_time, end_time, freq, future)
            MemCacheExpire.set_cache(H["c"], uri, result)

            get_module_logger("data").debug(f"从 {C.calendar_provider} 获取日历")
        else:
            get_module_logger("data").debug("从本地缓存获取日历")

        return result


# 全局内存缓存实例
H = MemCache()
