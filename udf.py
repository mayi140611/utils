#!/usr/bin/python
# encoding: utf-8
# '''
# 收集一些日常工作中编写过的实用函数

# * 
# '''
from functools import reduce
class udf(object):
    def __init__(self):
        pass
    
    def is_leap_year(self, year):
        '''
        判断是否是闰年
        '''
        try:
            year = int(year)
        except Exception as e:
            print(e)
            return

        if year % 100 == 0:
            if year % 400 == 0:
                return True
            else:
                return False
        else:
            if year % 4 == 0:
                return True
            else:
                return False

    def sort3num(self, x,y,z):
        '''
        输入三个整数x,y,z，请把这三个数由小到大输出。
        '''
        i = x
        if i>y:
            i=y
            y=x
            x=i
        if i>z:
            i=z
            z=y
            y=x
            x=i
        elif i<z and y>z:
            i=z
            z=y
            y=i
        return x,y,z

    
    def cal_factorial(self, num):
        '''
        计算num的阶乘。num为大于0的整数
        '''
        return reduce(lambda x,y: x*y, range(1,num+1))
        

    def cal_sum_factorial(self, num):
        '''
        计算1！+2！+...+num! 的结果。
        '''
        return sum([self.cal_factorial(i) for i in range(1, num+1)])
