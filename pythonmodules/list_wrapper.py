#!/usr/bin/python
# encoding: utf-8

#主要是对python中的list的相关操作的封装
import random

class list_wrapper(object):
    def __init__(self):
        pass
    @classmethod
    def sorted(self, iterable, key=None, reverse=False):
        '''
        Return a new list containing all items from the iterable in ascending order.
        @key: 
        A custom key function can be supplied to customize the sort order
            sorted(list1,key=abs)#[0, 1, -1, 2, 3, -3, 4, -4, -5]
        @reverse: 
        the reverse flag can be set to request the result in descending order.       
        '''
        return sorted(iterable, key, reverse)
    
    @classmethod
    def shuffle(self, x, random=None):
        '''
        Shuffle list x in place, and return None.

        Optional argument random is a 0-argument function returning a
        random float in [0.0, 1.0); if it is the default None, the
        standard random.random will be used. 
        '''
        return random.shuffle(x, random)