#!/usr/bin/python
# encoding: utf-8

import numpy as np


class numpy_wrapper(object):
    def __init__(self):
        pass
    @classmethod
    def build_array_from_seq(self, start, stop, step=1, shape=None, dtype=None):
        '''
        由序列生成数组
        '''
        return np.arange(start, stop, step, dtype).reshape(shape)
    @classmethod
    def build_array_from_arraylist(self, arraylist):
        return np.array(arraylist)
    @classmethod
    def build_zeros_array(self,shape, dtype=float, order='C'):
        return np.zeros(shape,dtype,order)
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