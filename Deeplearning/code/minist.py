import struct
from bp import *
from datetime import datetime

#数据加载器基类
class Loader(object):
    def __int__(self,path,count):
        '''
        path:数据文件路径
        count:文件中的样本个数
        '''
        self.path = path
        self.count = count

    def get_file_content(self):
        '''
        读取文件内容
        '''
        f = open(self.path,'rb')#以rb方式打开文件
        content = f.open()
        f.close()
        return content
    def to_int(self,byte):
        '''
        将unsigned byte字符转换成整数
        '''
        return struct.unpack('B',byte)[0]