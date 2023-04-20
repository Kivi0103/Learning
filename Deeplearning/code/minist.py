import struct
from bp import *
from datetime import datetime

#数据加载器基类
class Loader(object):
    def __init__(self,path,count):
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
        content = f.read()
        f.close()
        return content

    def to_int(self,byte):
        '''
        将unsigned byte字符转换成整数
        '''
        print(byte)
        return struct.unpack('B',byte)[0]#解包 将unsigned byte字符转换成整数

#图像数据加载器
class ImageLoader(Loader):#继承数据加载器基类
    def get_picture(self,content,index):
        """
        内部函数，从文件中获取图像
        """
        start = index*28*28+16   #+16什么意思
        picture = []
        for i in range(28):  #将28行像素加入picture列表
            picture.append([])
            for j in range(28): #每一行28个像素值
                picture[i].append(
                    content[start +i *28+j]
                )
        return picture #得到的picture是一个28行，28列的列表数据，每一行由28个数组成

    def get_one_sample(self,picture):
        """
        内部函数，将图像转化为样本的输入向量
        其实就是将上面get_picture函数得到的一个picture转换成一个1*784
        """
        sample = []
        for i in range(28):    #????感觉没什么作用呀，为什么不直接用上面的picture，或者直接在picture上就改成一个1*784的
            for j in range(28):
                sample.append(picture[i][j])
        return sample

    def load(self):
        """
        加载数据文件，获得全部样本的输入向量
        """
        content = self.get_file_content()
        data_set = []
        for index in range(self.count):
            data_set.append(
                self.get_one_sample(self.get_picture(content,index))
            )
        return data_set

#标签数据加载器
class LabelLoader(Loader):#继承了loader类
    def load(self):
        """
        加载数据文件，获得全部样本的标签向量
        """
        content = self.get_file_content()
        labels = []
        for index in range(self.count):
            labels.append(self.norm(content[index+8]))
        return labels

    def norm(self,label):
        """
        内部函数，将一个值转换为10维标签向量
        """
        label_vec = []
        label_value = label
        for i in range(10):  #one_hot化
            if i == label_value:
                label_vec.append(0.9)
            else:
                label_vec.append(0.1)
        return label_vec
def get_training_data_set():
    """
    获得训练数据集
    """
    image_loader = ImageLoader('D:/LearningGit/Learning/Deeplearning/resources/minist/train-images-idx3-ubyte', 60000)
    label_loader = LabelLoader('D:/LearningGit/Learning/Deeplearning/resources/minist/train-labels-idx1-ubyte', 60000)
    return image_loader.load(),label_loader.load()

def get_test_data_set():
    """
    获得测试数据集
    """
    image_loader = ImageLoader('D:/LearningGit/Learning/Deeplearning/resources/minist/t10k-images-idx3-ubyte',
                               10000)
    label_loader = LabelLoader('D:/LearningGit/Learning/Deeplearning/resources/minist/t10k-labels-idx1-ubyte',
                               10000)
    return image_loader.load(), label_loader.load()


def get_result(vec):
    max_value_index = 0
    max_value = 0
    for i in range(len(vec)):
        if vec[i] > max_value:
            max_value = vec[i]
            max_value_index = i
    return max_value_index


def evaluate(network,test_data_set,test_labels):
    error = 0
    total = len(test_data_set)
    for i in range(total):
        label = get_result(test_labels[i])
        predict = get_result(network.predict(test_data_set[i]))
        if label != predict:
            error += 1
    return float(error)/float(total)


def train_and_evaluate():
    last_error_ration = 1.0
    epoch = 0
    train_data_set,train_labels = get_training_data_set()
    test_data_set, test_labels = get_test_data_set()
    network = Network([784,300,10])
    while True:
        epoch += 1
        network.train(train_labels,train_data_set,0.3,1)
        print('%s epoch %d finished'%(datetime.datetime.now(),epoch))
        if epoch %10==0:
            error_ratio = evaluate(network, test_data_set, test_labels)
            print( '%s after epoch %d, error ratio is %f' % (datetime.datetime.now(), epoch, error_ratio))
            if error_ratio>last_error_ration:#本回合的错误率比上次还高 ？？？为什么不更新
                break
            else:#更新上次错误率
                last_error_ration = error_ratio


if __name__ == '__main__':
    train_and_evaluate()
