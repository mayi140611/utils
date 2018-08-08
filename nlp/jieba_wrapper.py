#!/usr/bin/python
# encoding: utf-8



import jieba


class jieba_wrapper(object):
    def __init(self):
        pass
    @classmethod
    def cut(self,sentence, cut_all=False, HMM=True):
        '''
        The main function that segments an entire sentence that contains
        Chinese characters into seperated words.
        :return: generator
        '''
        return jieba.cut(sentence, cut_all=cut_all, HMM=HMM)
    @classmethod
    def lcut(self,sentence, cut_all=False, HMM=True):
        '''
        The main function that segments an entire sentence that contains
        Chinese characters into seperated words.
        :return: list
        '''
        return jieba.lcut(sentence, cut_all=cut_all, HMM=HMM)
    @classmethod
    def cut_for_search(self,sentence, HMM=True):
        '''
        Finer segmentation for search engines.
        分词粒度较细，适合搜索引擎构建倒排索引
        :return: generator
        '''
        return jieba.cut_for_search(sentence, HMM=HMM)
    @classmethod
    def lcut_for_search(self,sentence, HMM=True):
        '''
        Finer segmentation for search engines.
        分词粒度较细，适合搜索引擎构建倒排索引
        :return: list
        '''
        return jieba.lcut_for_search(sentence, HMM=HMM)
    @classmethod
    def load_userdict(file_name):
        '''
        开发者可以指定自己自定义的词典，以便包含 jieba 词库里没有的词。
        虽然 jieba 有新词识别能力，但是自行添加新词可以保证更高的正确率
        词典格式和 dict.txt 一样，一个词占一行；
        每一行分三部分：词语、词频（可省略）、词性（可省略），用空格隔开，顺序不可颠倒。
        file_name 若为路径或二进制方式打开的文件，则文件必须为 UTF-8 编码。
        词频省略时使用自动计算的能保证分出该词的词频。
        '''
        return jieba.lcut_for_search(sentence, HMM=HMM)
        
        