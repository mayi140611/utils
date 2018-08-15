#!/usr/bin/python
# encoding: utf-8

'''
数据分析各个阶段的wrapper
* 数据探索（包含数据预处理过程）
数据编码、缺失值处理、异常值处理、字段间相关性分析、数值型字段描述性统计、
* 数据探索结果可视化
* 特征工程
* 数据集分拆
* 模型构建
* 
'''
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.metrics.pairwise import pairwise_distances
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split

class sklearn_wrapper(object):
    def __init__(self):
        pass
    def train_test_split_df(self, df, test_size=0.25):
        '''
        把df切分成测试集和训练集。在切分前会打乱数据集
        return: train_data, test_data = train_test_split(df, test_size)
        '''
        return train_test_split(df, test_size)
    def cosine_similarity(self, X, Y=None, dense_output=True):
        '''
        Compute cosine similarity between samples in X and Y. 
        If Y == ``None``, the output will be the pairwise
        similarities between all samples(每个sample即X的一行) in ``X``.
        @X : ndarray or sparse array,
        @Y : ndarray or sparse array,
        @return ndarray。元素值在(-1,1)
        '''
        return cosine_similarity(X, Y)
    def cosine_distances(self,X, Y=None):
        '''
        cosine_distances = 1-cosine_similarity
        Compute cosine distances距离 between samples in X and Y. 
        If Y == ``None``, the output will be the pairwise
        distances between all samples(每个sample即X的一行) in ``X``.
        @X : ndarray or sparse array,
        @Y : ndarray or sparse array,
        @return ndarray。元素值在(-1,1)
        '''
        return pairwise_distances(X,Y, metric='cosine')
    def mean_squared_error(self,y_true, y_pred, sample_weight=None, multioutput='uniform_average'):
        '''
        计算均方误差。Mean squared error regression loss
        @y_true：一维ndarray
        '''
        return mean_squared_error(y_true, y_pred, sample_weight, multioutput)
        