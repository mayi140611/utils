from pyquery import PyQuery as pq

class pyquery_wrapper(object):
    def __init__(self, text):
        self._root = pq(text)

    @property
    def Root(self):
        '''
        返回文档的根
        '''
        return self._root