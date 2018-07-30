#!/usr/bin/python
# encoding: utf-8
import pandas as pd

class pandas_wrapper(object):
    def __init__(self):
        pass

    def build_df_with_dict(self, data=None, index=None, columns=None, dtype=None, copy=False):
        '''
        data = {'state': ['Ohio', 'Ohio', 'Ohio', 'Nevada', 'Nevada'],
            'year': [2000, 2001, 2002, 2001, 2002],
            'pop': [1.5, 1.7, 3.6, 2.4, 2.9]}
        data = {'Nevada': {2001: 2.4, 2002: 2.9},
            'Ohio': {2000: 1.5, 2001: 1.7, 2002: 3.6}}
        '''
        return pd.DataFrame(data)

# 索引数据
    def loc(self,df,indexes, variables):
        '''
        通过索引和列名获取数据
        indexes: 索引列表
        variables：变量名列表
        '''
        return df.loc[indexes,variables]
    def iloc(self,df,rows, columns):
        '''
        通过索引和列名获取数据
        rows: 行号列表
        columns：列号列表
        '''
        return df.iloc[indexes,columns]