#!/usr/bin/python
# encoding: utf-8
import pandas as pd


class dataframe_wrapper(object):
    def __init__(self):
        pass
    
    #DataFrame
    # 索引数据
    @classmethod
    def loc(self,df,rnames, cnames):
        '''
        通过索引和列名获取数据
        rnames: 行名（索引）列表
        cnames：列名（变量名）列表
        '''
        return df.loc[rnames,cnames]
    @classmethod
    def iloc(self,df,rnums, cnums):
        '''
        通过索引和列名获取数据
        rnums: 行号列表
        cnums：列号列表
        '''
        return df.iloc[rnums,cnums]
    @classmethod
    def get_not_null_df(self,df,cname):
        '''
        获取df中某列不为空的全部数据
        cname: string
        '''
        return df[pd.notnull(df[cname])]
    @classmethod
    def sort_by_column(self,df,cname,axis=0, ascending=True, inplace=False, kind='quicksort', na_position='last'):
        '''
        按照df的某一列的值进行排序
        cname: string
        '''
        return df.sort_values(by=cname,axis=axis, ascending=ascending, inplace=inplace, kind=kind, na_position=na_position)
    @classmethod
    def get_unique_col_values(self, df,cname):
        '''
        返回df的cname列的唯一值
        cname: string
        return: The unique values returned as a NumPy array. In case of categorical
        data type, returned as a Categorical.
        '''
        return df[cname].unique()
    
    @classmethod
    def itertuples(self, df):
        '''
        获取df的迭代器，按行迭代，每行是一个<class 'pandas.core.frame.Pandas'>对象，可以当做元组看待
        如：Pandas(Index=13286, user_id=5, item_id=385, rating=4, timestamp=875636185)
        for line in df.itertuples():
            #由于user_id和item_id都是从1开始编号的，所有要减一
            train_data_matrix[line[1], line[2]] = line[3] 
        line[0]是df的索引
        '''
        return df.itertuples()