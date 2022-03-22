# Custom cache instance class must implement AbstractCacheInstance interface:
from easy_cache import caches
from easy_cache import ecached, ecached_property, create_cache_key
from easy_cache.abc import AbstractCacheInstance
from easy_cache.core import DEFAULT_TIMEOUT, NOT_FOUND
from .util import data_dir
from os.path import join
from os import listdir, unlink
import pickle as pickle
from photospline import SplineTable
import gzip
import json
import time


class PickleCache(AbstractCacheInstance):
    def __init__(self, base_dir=join(data_dir, "cache"), *args, **kwargs):
        super(PickleCache, self).__init__(*args, **kwargs)
        self._base_dir = base_dir
        try:
            with open(join(self._base_dir, "manifest.json")) as f:
                self._manifest = json.load(f)
        except IOError:
            self._manifest = {}
        self._sweep()

    def _get_filename(self, key):
        return join(self._base_dir, self._manifest[key]["filename"])

    def _dump_manifest(self):
        with open(join(self._base_dir, "manifest.json"), "w") as f:
            json.dump(self._manifest, f, indent=1)

    def _dump_item(self, key, value):
        """
        Special-case serialization for SplineTable
        """
        if isinstance(value, SplineTable):
            fname = key + ".fits"
            value.write(join(self._base_dir, fname))
        else:
            fname = key + ".pkl.gz"
            with gzip.open(join(self._base_dir, fname), "wb") as f:
                pickle.dump(value, f, protocol=pickle.HIGHEST_PROTOCOL)
        return fname

    def _load_item(self, fname):
        if fname.endswith(".fits"):
            return SplineTable(fname)
        elif fname.endswith(".pkl.gz"):
            with gzip.open(fname, "r") as f:
                return pickle.load(f)
        else:
            raise ValueError("Don't know how to load {}".format(fname))

    def _sweep(self):
        removed = set()
        for key, item in self._manifest.items():
            if item["expires"] is not None and item["expires"] <= time.time():
                try:
                    unlink(join(self._base_dir, item["filename"]))
                except FileNotFoundError:
                    ...
                removed.add(key)
        for k in removed:
            del self._manifest[k]
        self._dump_manifest()

    def get(self, key, default=NOT_FOUND):
        if not key in self._manifest:
            return default
        else:
            try:
                return self._load_item(self._get_filename(key))
            except EOFError:
                self.delete(key)
                return default

    def get_many(self, keys):
        return [self.get(k) for k in keys]

    def set(self, key, value, timeout=DEFAULT_TIMEOUT):
        if timeout == DEFAULT_TIMEOUT:
            expires = None
        else:
            expires = time.time() + timeout

        filename = self._dump_item(key, value)
        self._manifest[key] = {"filename": filename, "expires": expires}
        self._dump_manifest()

    def set_many(self, data_dict, timeout=DEFAULT_TIMEOUT):
        if timeout == DEFAULT_TIMEOUT:
            expires = None
        else:
            expires = time.time() + timeout
        for key, value in data_dict.items():
            filename = self._dump_item(key, value)
            self._manifest[key] = {"filename": filename, "expires": expires}

        self._dump_manifest()

    def delete(self, key):
        try:
            filename = self._get_filename(key)
        except KeyError:
            return
        try:
            unlink(filename)
            del self._manifest[key]
        except (OSError, KeyError):
            pass


caches.set_default(PickleCache())

try:
    from functools import lru_cache
except ImportError:
    from backports.functools_lru_cache import lru_cache
