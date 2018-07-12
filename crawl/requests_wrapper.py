import requests

class requests_wrapper(object):
    headers = {'user-agent': 'Mozilla/5.0 (Windows NT 10.0; WOW64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/61.0.3163.79 Safari/537.36 Maxthon/5.2.1.6000'}
    def __init__(self, url):
        self._url = url

    def get_response(self, encoding='utf8', data=None):
        r = requests.get(self._url,headers=self.headers)
        r.encoding=encoding
        return r
    
    def get_text(self, encoding='utf8', data=None):
        return self.get_response(encoding, data).text
    