#!/usr/bin/python
# encoding: utf-8

import numpy as np


class numpy_wrapper(object):
    def __init__(self):
        pass
    @classmethod
    def arange(self, start, stop, step=1, dtype=None):
        '''
        生成1D ndarray
        Return evenly spaced values within a given interval.
        '''
        return np.arange(start, stop, step, dtype)
    @classmethod
    def linspace(self, start, stop, num=50, endpoint=True, retstep=False, dtype=None):
        '''
        生成1D ndarray
        Return evenly spaced numbers over a specified interval.
        '''
        return np.linspace(start, stop, num, endpoint, retstep, dtype)
    @classmethod
    def build_array_from_seq(self, start, stop, step=1, shape=None, dtype=None):
        '''
        由序列生成数组
        @shape: list or tuple
        '''
        return np.arange(start, stop, step, dtype).reshape(shape)
    @classmethod
    def build_array_from_arraylist(self, arraylist):
        return np.array(arraylist)
    @classmethod
    def build_zeros_array(self,shape, dtype=float, order='C'):
        return np.zeros(shape,dtype,order)
    
    @classmethod
    def add_newaxis_last(self,matr):
        '''
        在matr的最后加一个维度
        @matr: 1D ndarray
        '''
        return matr[:,np.newaxis]
    @classmethod
    def flatten(self,order='C'):
        '''
        Return a copy of the array collapsed(坍塌) into one dimension.
        @order: {'C', 'F', 'A', 'K'}, optional
        'C' means to flatten in row-major (C-style) order.
        'F' means to flatten in column-major (Fortran-
        style) order. 'A' means to flatten in column-major
        order if `a` is Fortran *contiguous* in memory,
        row-major order otherwise. 'K' means to flatten
        `a` in the order the elements occur in memory.
        The default is 'C'.
        '''
        return np.flatten(order)
    # ---------------------------------------------------------------------------------
    # 描述性统计
    # ---------------------------------------------------------------------------------
    
    @classmethod
    def max(self, a, axis=None):
        '''
        求最大值
        @axis: None表示求整个数组的最大值；0表示求每列最大值；1表示求每行最大值
        '''
        return np.max(a, axis)
    @classmethod
    def min(self, a, axis=None):
        '''
        求最大值
        @axis: None表示求整个数组的最大值；0表示求每列最大值；1表示求每行最大值
        '''
        return np.min(a, axis)
    @classmethod
    def sum(self, a, axis=None):
        '''
        求最大值
        @axis: None表示求整个数组的最大值；0表示求每列最大值；1表示求每行最大值
        '''
        return np.sum(a, axis)
    @classmethod
    def mean(self, a, axis=None):
        '''
        求最大值
        @axis: None表示求整个数组的最大值；0表示求每列最大值；1表示求每行最大值
        '''
        return np.mean(a, axis)
    # ---------------------------------------------------------------------------------
    # 生成随机数
    # ---------------------------------------------------------------------------------
    @classmethod
    def generate_random_seed(self, seed=None):
        '''
        但是numpy.random.seed()不是线程安全的，
        如果程序中有多个线程最好使用numpy.random.RandomState实例对象来创建
        或者使用random.seed()来设置相同的随机数种子。
                
        import random
        random.seed(1234567890)
        a = random.sample(range(10),5) 

        注意： 随机数种子seed只有一次有效，在下一次调用产生随机数函数前没有设置seed，则还是产生随机数。
        '''
        return np.random.RandomState(seed)
    @classmethod
    def uniform_rand(self, *param, seed=None):
        '''
        Create an array of the given shape and populate it with
        random samples from a uniform distribution
        over ``[0, 1)``.
        '''
        return self.generate_random_seed(seed).rand(*param)
    @classmethod
    def uniform_randint(self, low, high=None, size=None, dtype='l', seed=None):
        '''
        Return random integers from `low` (inclusive) to `high` (exclusive).
        '''
        return self.generate_random_seed(seed).randint(low, high, size, dtype)
    @classmethod
    def randn(self, *param, seed=None):
        '''
        Return a sample (or samples) from the "standard normal" distribution.
        '''
        return self.generate_random_seed(seed).randn(*param)
    # '''
    # ---------------------------------------------------------------------------------
    # 线性代数
    # ---------------------------------------------------------------------------------
    # '''
    @classmethod
    def inv(self, matr):
        '''
        求解矩阵的逆
        '''
        return np.linalg.inv(matr)
    @classmethod
    def det(self, matr):
        '''
        计算行列式
        '''
        return np.linalg.det(matr)
    @classmethod
    def sum(self, matr, axis=None, dtype=None, out=None):
        return np.sum(matr, axis, dtype, out)