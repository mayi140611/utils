#!/usr/bin/python
# encoding: utf-8
# version: pymongo==3.7.1

#待处理：
#db.getCollection('symptomsdetail').find({}).sort({'titlepy':1})
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
            注：把不存在某个属性的行都查出来的条件{'c2':{'$exists':False}}
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

    def find_one(self, collection, conditions=None, fieldlist='all'):
        '''
        查找所有数据，返回指定的fieldlist
        :conditions 查询条件。{'c1':'全身'}
            注：把不存在某个属性的行都查出来的条件{'c2':{'$exists':False}}
        :fieldlist 'all'表示返回所有数据，或者是一个字段list
        '''
        d = dict()
        if fieldlist != 'all':
            if '_id' not in fieldlist:
                d['_id'] = 0
            for i in fieldlist:
                d[i] = 1
            return collection.find_one(conditions, d)
        return collection.find_one(conditions)

    def update_doc(self, collection, conditions,fielddict):
        '''
        更新表中的符合条件的第一条记录。注意：如果field存在，则更新值，如果不存在，则新增；但是不能删除field
        :conditions:如{'c1':'v1','c2':'v2'}
        :fielddict: 如{'f1':'v1','f2':'v2'}
        '''
        return collection.update_one(conditions,{ "$set": fielddict})
    def update_docs(self, collection, conditions,fielddict):
        '''
        更新表中符合条件的所有记录。注意：如果field存在，则更新值，如果不存在，则新增；但是不能删除field
        :conditions:如{'c1':'v1','c2':'v2'}
        :fielddict: 如{'f1':'v1','f2':'v2'}
        '''
        return collection.update_many(conditions,{ "$set": fielddict})
    
    def remove_doc_fields(self, collection, conditions,fieldslist):
        '''
        更新表中符合条件的一条记录。注意：如果field存在，则更新值，如果不存在，则新增；但是不能删除field
        :fieldslist: 如['f1','f2'...]
        '''
        return collection.update_one(conditions,{ "$unset": {ii:"" for ii in fieldslist}})
    def remove_docs_fields(self, collection, conditions,fieldslist):
        '''
        更新表中符合条件的所有记录。注意：如果field存在，则更新值，如果不存在，则新增；但是不能删除field
        :fieldslist: 如['f1','f2'...]
        '''
        return collection.update_many(conditions,{ "$unset": {ii:"" for ii in fieldslist}})