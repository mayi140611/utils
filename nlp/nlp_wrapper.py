#!/usr/bin/python
# encoding: utf-8
'''
主要有如下任务：
* word2vec
* character2vec
* 计算word相似度

'''
import distance

class nlp_wrapper(object):
    def __init__(self):
        pass
    '''
    ----------------------------------------------------------------------------------------------
    计算word相似度
    * 根据编辑距离
    * 根据word2vec
    ----------------------------------------------------------------------------------------------
    '''
    @classmethod
    def cal_editdistance(self, s1, s2, mode=1):
        '''
        通过编辑距离计算字符串相似度，可以是任意字符串word，sentence，text

        :mode: 计算相似的算法选择
        '''
        if mode ==1:
            return distance.levenshtein(s1, s2)
        elif mode ==2:
            matrix = [[i+j for j in range(len(str2) + 1)] for i in range(len(str1) + 1)]  

            for i in range(1,len(str1)+1):  
                for j in range(1,len(str2)+1):  
                    if str1[i-1] == str2[j-1]:  
                        d = 0  
                    else:  
                        d = 1  
                    matrix[i][j] = min(matrix[i-1][j]+1,matrix[i][j-1]+1,matrix[i-1][j-1]+d)  

            return matrix[len(str1)][len(str2)]  

    
#     @classmethod
#     def cal_editdistance(self, s1, s2, mode=1):  