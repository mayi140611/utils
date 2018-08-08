#!/usr/bin/python
# encoding: utf-8

from pypinyin import Style, pinyin,lazy_pinyin

class pypinyin_wrapper(object):
    def __init__(self):
        pass
    
    def  acronym(self, s1):
        '''
        获取汉字短语的首字母缩写，如下雨天，xyt
        '''
        return ''.join([ii[0] for ii in pinyin(s1, style=Style.FIRST_LETTER, strict=False)])
    
    def lazy_pinyin(self,s1):
        '''
        汉字短语转化为拼音列表，如'中心'，['zhong', 'xin']
        '''
        return lazy_pinyin(s1)