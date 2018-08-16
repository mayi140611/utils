#!/usr/bin/python
# encoding: utf-8
import pandas as pd

class series_wrapper(object):
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
    
    @classmethod
    def nonzero(self,series):
        '''
        Return the *integer* indices of the elements that are non-zero
        >>> s = pd.Series([0, 3, 0, 4], index=['a', 'b', 'c', 'd'])
        # same return although index of s is different
        >>> s.nonzero()
        (array([1, 3]),)
        >>> s.iloc[s.nonzero()[0]]
        b    3
        d    4
        dtype: int64
        '''
        return series.nonzero()
    @classmethod
    def nonzero_item(self,series):
        '''
        Return series中的非零元素
        >>> s = pd.Series([0, 3, 0, 4], index=['a', 'b', 'c', 'd'])
        # same return although index of s is different
        >>> s.nonzero()
        (array([1, 3]),)
        >>> s.iloc[s.nonzero()[0]]
        b    3
        d    4
        dtype: int64
        '''
        return series.iloc[series.nonzero()[0]]