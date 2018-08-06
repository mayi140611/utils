#!/usr/bin/python
# encoding: utf-8
from pymongo import MongoClient

class pymongo_wrapper(object):
    def __init__(self):
        self._client = MongoClient()

    def get_db(self, dbname):
        return self._client[dbname]

    def get_collection(self, dbname, collection):
        return self.get_db(dbname)[collection]
    
    def set_unique_index(self, dbname, collection, field):
        '''
        为单一字段设置唯一索引
        '''
        try:
            self.get_db(dbname)[collection].ensure_index(field, unique=True )
            return True
        except Exception as e:
            print(e)
            return False
    def find_all(self, collection, conditions=None, fieldlist='all'):
        '''
        查找所有数据，返回指定的fieldlist
        :conditions 查询条件。{'c1':'全身'}
        :fieldlist 'all'表示返回所有数据，或者是一个字段list
        '''
        d = dict()
        if fieldlist != 'all':
            if '_id' not in fieldlist:
                d['_id'] = 0
            for i in fieldlist:
                d[i] = 1
            return collection.find(conditions, d)
        return collection.find(conditions)

