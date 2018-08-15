#!/usr/bin/python
# encoding: utf-8
import pandas as pd

class pandas_wrapper(object):
    def __init__(self):
        pass
    @classmethod
    def read_csv(self,filepath_or_buffer, sep=',', delimiter=None, header='infer', names=None, index_col=None, usecols=None):
        '''
        Read CSV (comma-separated) file into DataFrame
        '''
        return pd.Series(data, index, dtype, name, copy, fastpath)
    @classmethod
    def build_series(self,data=None, index=None, dtype=None, name=None, copy=False, fastpath=False):
        '''
        Series:One-dimensional ndarray with axis labels (including time series).
        :data: 可以是list or dict
        '''
        return pd.Series(data, index, dtype, name, copy, fastpath)

    @classmethod
    def build_series_from_dict(self,data=None, index=None, dtype=None, name=None, copy=False, fastpath=False):
        '''
        Series:One-dimensional ndarray with axis labels (including time series).
        '''
        return pd.Series(data, index, dtype, name, copy, fastpath)
