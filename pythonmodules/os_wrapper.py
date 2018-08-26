#!/usr/bin/python
# encoding: utf-8

#主要是对python中的os的相关操作的封装

import os

class os_wrapper(object):
    def __init__(self):
        pass
    @classmethod
    def abspath(self,path):
        '''
        获取绝对路径
        >>>os.path.abspath('.')
        '/home/ian/code/github/utils/0examples'
        '''
        return os.path.abspath(path)
    
    @classmethod
    def dirname(self,path):
        '''
        获取路径所在的文件夹
        >>>os.path.dirname('ian/code/github/utils/0examples/')
        'ian/code/github/utils/0examples'
        >>>os.path.dirname('ian/code/github/utils/0examples')
        'ian/code/github/utils'
        '''
        return os.path.dirname(path)
    
    @classmethod
    def join(self,a, *p):
        '''
        Join two or more pathname components, inserting '/' as needed.
        If any component is an absolute path, all previous path components
        will be discarded.  An empty last part will result in a path that
        ends with a separator.
        >>>os.path.join('ian/code/','github/utils/0examples')
        'ian/code/github/utils/0examples'
        >>>os.path.join('ian/code','github/utils/0examples')
        'ian/code/github/utils/0examples'
        >>>os.path.join('ian/code','/github/utils/0examples')
        '/github/utils/0examples'
        '''
        return os.path.join(a, *p)