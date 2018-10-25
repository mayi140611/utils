#!/usr/bin/python
# encoding: utf-8
import pandas as pd


class dataframe_wrapper(object):
    def __init__(self):
        pass
    
    '''
    #####################################
    DataFrame索引数据
    #####################################
    '''
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
    def rename(df, index=None, columns=None, copy=True, inplace=False, level=None):
        '''
        修改df的index和columns的名称，默认会返回一个新的DF
        Alter axes labels.

        Function / dict values must be unique (1-to-1). Labels not contained in
        a dict / Series will be left as-is（被遗弃）. Extra labels listed don't throw an
        error.
        >>> df4.rename(index={0:'01'})#修改索引名
        >>> concept.rename(columns={'code':'symbol','c_name':'concept'})#修改列名
        Parameters
        ----------
        index, columns : dict-like or function, optional
            dict-like or functions transformations to apply to
            that axis' values. Use either ``mapper`` and ``axis`` to
            specify the axis to target with ``mapper``, or ``index`` and
            ``columns``.
        copy : boolean, default True
            Also copy underlying data
        inplace : boolean, default False
            Whether to return a new DataFrame. If True then value of copy is
            ignored.
        level : int or level name, default None
            In case of a MultiIndex, only rename labels in the specified
            level.

        Returns
        -------
        renamed : DataFrame

        '''
        return df.rename(mapper=None, index=index, columns=columns, axis=None, copy=copy, inplace=inplace, level=level)
        
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
    '''
    #####################################
    DataFrame合并
    #####################################
    '''
    def merge(left, right, how='inner', on=None, left_on=None, right_on=None, left_index=False, right_index=False, sort=False, suffixes=('_x', '_y'), copy=True, indicator=False, validate=None):
        '''
        按照两个df相同的列名合并DataFrame
        Merge DataFrame objects by performing a database-style join operation by
        columns or indexes.
        Parameters
        ----------
        left : DataFrame
        right : DataFrame
        how : {'left', 'right', 'outer', 'inner'}, default 'inner'
            * left: use only keys from left frame, similar to a SQL left outer join;
              preserve key order
            * right: use only keys from right frame, similar to a SQL right outer join;
              preserve key order
            * outer: use union of keys from both frames, similar to a SQL full outer
              join; sort keys lexicographically
            * inner: use intersection of keys from both frames, similar to a SQL inner
              join; preserve the order of the left keys
        on : label or list
            根据两个df共有的列名进行合并
            Column or index level names to join on. These must be found in both
            DataFrames. If `on` is None and not merging on indexes then this defaults
            to the intersection of the columns in both DataFrames.
        left_on : label or list, or array-like
            如果根据两个df的不同的列名合并，则需要制定left_on&right_on
            Column or index level names to join on in the left DataFrame. Can also
            be an array or list of arrays of the length of the left DataFrame.
            These arrays are treated as if they are columns.
        right_on : label or list, or array-like
            Column or index level names to join on in the right DataFrame. Can also
            be an array or list of arrays of the length of the right DataFrame.
            These arrays are treated as if they are columns.
        left_index : boolean, default False
            也可以根据index进行合并
            Use the index from the left DataFrame as the join key(s). If it is a
            MultiIndex, the number of keys in the other DataFrame (either the index
            or a number of columns) must match the number of levels
        right_index : boolean, default False
            Use the index from the right DataFrame as the join key. Same caveats as
            left_index     
        sort : boolean, default False
            Sort the join keys lexicographically in the result DataFrame. If False,
            the order of the join keys depends on the join type (how keyword)
        suffixes : 2-length sequence (tuple, list, ...)
            如果合并的两个的df有相同的列名，则加后缀区分
            Suffix to apply to overlapping column names in the left and right
            side, respectively
        copy : boolean, default True
            If False, do not copy data unnecessarily
        indicator : boolean or string, default False
            If True, adds a column to output DataFrame called "_merge" with
            information on the source of each row.
            If string, column with information on source of each row will be added to
            output DataFrame, and column will be named value of string.
            Information column is Categorical-type and takes on a value of "left_only"
            for observations whose merge key only appears in 'left' DataFrame,
            "right_only" for observations whose merge key only appears in 'right'
            DataFrame, and "both" if the observation's merge key is found in both.

        validate : string, default None
            If specified, checks if merge is of specified type.

            * "one_to_one" or "1:1": check if merge keys are unique in both
              left and right datasets.
            * "one_to_many" or "1:m": check if merge keys are unique in left
              dataset.
            * "many_to_one" or "m:1": check if merge keys are unique in right
              dataset.
            * "many_to_many" or "m:m": allowed, but does not result in checks.
        '''
        return pd.merge(left, right, how=how, on=on, left_on=left_on, right_on=right_on, left_index=left_index, right_index=right_index, sort=sort, suffixes=suffixes, copy=copy, indicator=indicator, validate=validate)
    def join(left, other, on=None, how='left', lsuffix='', rsuffix='', sort=False):
        '''
        和merge函数类似，只不过调用的主体是left_df
        '''
        return left.join(other, on=None, how='left', lsuffix='', rsuffix='', sort=False)
    def concat(objs, axis=0, join='outer', join_axes=None, ignore_index=False, keys=None, levels=None, names=None, verify_integrity=False, sort=None, copy=True):
        '''
        如果axis=0，按照相同的column名进行列方向的叠加
        如果axis=1，按照相同的index名进行行方向的叠加
        '''
        return None
        