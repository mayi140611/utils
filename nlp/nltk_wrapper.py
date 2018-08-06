#!/usr/bin/python
# encoding: utf-8
from nltk.util import trigrams
from nltk.util import bigrams
# nltk中有以下几个概念
# raw：原始文本，可以看作一个大的string
# text
# words：word list。[w1,w2,w3...]
# sents：句子list。以'. '和换行符'\n'等作为分隔符的词链表。如下形式：
#     [[w1,w2...],[w3,w4,...],...]
# 相互转换
# raw转text: 需要先转成words list, 如 nltk.Text(raw.split())
# raw转words list：raw.split()
# raw转sents list: ?
# text转raw：''.join(list(text))
# text转words list: list(text)
# text转sents：？
# words转raw：''.join(words)
# words转text: nltk.Text(words)
# words转sents: ?
# sents转words: [w for s in sents for w in s]
# sents转raw
# sents转text


class nltk_wrapper(object):
    def __init__(self, text):
        '''
        text: 一个nltk.text.Text类型的语料
        '''
        self._text = text

#文本探索
#输入一个nltk.text.Text类型的语料，进行一些统计分析
    @property
    def length(self):
        '''
        返回文本的总长度（不包含空格）
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
    def words_freq(self, n):
        '''
        返回text的词频字典
        List the n most common elements and their counts from the most
        common to the least.  If n is None, then list all element counts.
        等价于nltk.book.FreqDist(self._text).most_common()
        '''        
        return self._text.vocab().most_common(n)
    
    def lexical_diversity(self): 
        '''
        返回重复词密度：平均每个词重复几次
        '''        
        return len(self._text) / len(set(self._text)) 
    
    def percentage(self, word): 
        '''
        返回关键词密度
        '''        
        return self.countword(word)/self.length
    
#文本处理
    def get_bigrams(self):
        return bigrams(self._text)
    def get_trigrams(self):
        return trigrams(self._text)
    def get_words(self, length):
        '''
        返回text中所有length长度的word
        '''        
        return [word for word in self._text if len(word)==length]
#载入自己的语料库
    def load_corpus(self, corpus_root, regex='.*', encoding='utf8'):
        '''
        返回自己的语料库
        corpus_root：语料所在的目录
        regex：正则，默认导入目录下所有文件
        '''        
        from nltk.corpus import PlaintextCorpusReader
        wordlists = PlaintextCorpusReader(corpus_root, regex, encoding= encoding)
        return wordlists