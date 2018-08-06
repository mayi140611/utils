#!/usr/bin/python
# encoding: utf-8

#主要是对python中的re的相关操作的封装

import re
import nltk

class re_wrapper(object):
    def __init__(self):
        pass

    def re_show(self, regexp, string, left='{', right='}'):
        '''
        把找到的符合regexp的non-overlapping matches标记出来
        如：
        nltk.re_show('[a-zA-Z]+','12fFdsDFDS3rtG4')#12{fFdsDFDS}3{rtG}4
        '''
        return nltk.re_show(regexp, string, left, right)

    def findall(self,regexp, string):
        '''
        如果regexp中不包含小括号，如
        re.findall('[a-zA-Z]+','12fFdsDFDS3rtG4')#['fFdsDFDS', 'rtG']
        等价于re.findall('([a-zA-Z]+)','12fFdsDFDS3rtG4')#['fFdsDFDS', 'rtG']
        否则：
        re.findall('(\d)\s+(\d)','12 3fFdsDFDS3 4rtG4')#[('2', '3'), ('3', '4')]
        :return: list
        '''
        return re.findall(regexp, string)