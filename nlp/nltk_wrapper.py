#!/usr/bin/python
# encoding: utf-8
from nltk.util import trigrams
from nltk.util import bigrams


class nltk_wrapper(object):
    def __init__(self, text):
        self._text = text

#文本探索
#输入一个nltk.text.Text的语料，进行一些统计分析
    @property
    def length(self):
        '''
        返回文本的总长度
        '''
        return len(self._text)
    def vocabulary(self):
        '''
        按照字母升序返回文本的词汇列表
        '''
        return sorted(set(self._text))
    def countword(self, word):
        '''
        返回某个word在text中出现的次数
        '''
        return self._text.count(word)
    def words_freq(self):
        '''
        返回text的词频字典
        等价于nltk.book.FreqDist(self._text)
        '''        
        return self._text.vocab()

#文本处理
    def get_bigrams(self):
        return bigrams(self._text)
    def get_trigrams(self):
        return trigrams(self._text)