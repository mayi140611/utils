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
    def build_df_from_dict(self, data=None, index=None, columns=None, dtype=None, copy=False):
        '''
        data = {'state': ['Ohio', 'Ohio', 'Ohio', 'Nevada', 'Nevada'],
            'year': [2000, 2001, 2002, 2001, 2002],
            'pop': [1.5, 1.7, 3.6, 2.4, 2.9]}
        data = {'Nevada': {2001: 2.4, 2002: 2.9},
            'Ohio': {2000: 1.5, 2001: 1.7, 2002: 3.6}}
        '''
        return pd.DataFrame(data)
    @classmethod
    def series_inverse(self, series):
        '''
        把series的index和values互换
        '''
        return pd.Series(series.index,index=series.values)
    