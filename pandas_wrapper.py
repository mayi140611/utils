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
#DataFrame
# 索引数据
    def loc(self,df,rnames, cnames):
        '''
        通过索引和列名获取数据
        rnames: 行名（索引）列表
        cnames：列名（变量名）列表
        '''
        return df.loc[rnames,cnames]
    def iloc(self,df,rnums, cnums):
        '''
        通过索引和列名获取数据
        rnums: 行号列表
        cnums：列号列表
        '''
        return df.iloc[rnums,cnums]

    def get_not_null_df(self,df,cname):
        '''
        获取df中某列不为空的全部数据
        cname: string
        '''
        return df[pd.notnull(df[cname])]
