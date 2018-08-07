import pickle

class pickle_wrapper(object):
    def __init__(self):
        pass
    @classmethod
    def loadfromfile(self, file, mode='rb'):
        with open(file, mode) as f:
            return pickle.load(f)
    
    @classmethod
    def dump2file(self, o, file, mode='wb'):
        with open(file, mode) as f:
            pickle.dump(o,f)