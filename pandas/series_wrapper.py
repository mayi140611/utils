#!/usr/bin/python
# encoding: utf-8
import pandas as pd

class pandas_wrapper(object):
    def __init__(self):
        pass
    @classmethod
    def append(self,to_append, ignore_index=False, verify_integrity=False):
        '''
        Concatenate two or more Series.
        注意这里不同于list.append, 相当于list.extend
        to_append : Series or list/tuple of Series
        ignore_index : boolean, default False
            If True, do not use the index labels.
        verify_integrity : boolean, default False
            If True, raise Exception on creating index with duplicates
        '''
        return pd.Series(data, index, dtype, name, copy, fastpath)