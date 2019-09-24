# Custom cache instance class must implement AbstractCacheInstance interface:
from easy_cache.abc import AbstractCacheInstance
from easy_cache.core import DEFAULT_TIMEOUT, NOT_FOUND
from .util import data_dir
from os.path import join
from os import listdir, unlink
import cPickle as pickle
import gzip
import time

class PickleCache(AbstractCacheInstance):

    def __init__(self, base_dir=join(data_dir, 'cache'), *args, **kwargs):
        super(PickleCache, self).__init__(*args, **kwargs)
        self._base_dir = base_dir
        self._sweep()

    def _get_filename(self, key):
        return join(self._base_dir, key+'.pkl.gz')
    
    def _sweep(self):
        cache_dir = join(data_dir, 'cache')
        for fname in listdir(self._base_dir):
            fname = join(self._base_dir, fname)
            if fname.endswith('.pkl.gz'):
                with gzip.open(fname, 'r') as f:
                    try:
                        item = pickle.load(f)
                    except EOFError:
                        unlink(fname)
                        continue
                if item['expires'] is not None and item['expires'] <= time.time():
                    unlink(fname)

    def get(self, key, default=NOT_FOUND):
        try:
            try:
                with gzip.open(self._get_filename(key), 'r') as f:
                    item = pickle.load(f)
            except EOFError:
                self.delete(key)
                return default
            if item['expires'] is not None and item['expires'] <= time.time():
                self.delete(key)
                return default
            else:
                return item['value']
        except IOError:
            return default

    def get_many(self, keys):
        return [self.get(k) for k in keys]

    def set(self, key, value, timeout=DEFAULT_TIMEOUT):
        if timeout == DEFAULT_TIMEOUT:
            expires = None
        else:
            expires = time.time() + timeout
        with gzip.open(self._get_filename(key), 'wb') as f:
            pickle.dump({'key': key, 'expires': expires, 'value': value}, f, protocol=pickle.HIGHEST_PROTOCOL)

    def set_many(self, data_dict, timeout=DEFAULT_TIMEOUT):
        for k, v in data_dict.items():
            self.set_many(k,v,timeout)

    def delete(self, key):
        unlink(self._get_filename(key))

from easy_cache import caches
caches.set_default(PickleCache())

from easy_cache import ecached, ecached_property, create_cache_key
try:
    from functools import lru_cache
except ImportError:
    from backports.functools_lru_cache import lru_cache