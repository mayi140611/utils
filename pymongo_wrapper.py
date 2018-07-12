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