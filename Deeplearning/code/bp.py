import random
from functools import reduce

from numpy import *

# 定义sigmoid激活函数
def sigmoid(inX):
    return 1.0/(1+exp(-inX))

# 节点类，负责记录和维护节点自身信息以及与这个节点相关的上下游链接，实现输出值和误差项的计算
class Node(object):
    def __init__(self,layer_index,node_index):
        '''
        构造节点对象。
        '''
        self.layer_index = layer_index #节点所属的层的编号
        self.node_index = node_index #节点的编号
        self.downstream = [] #下游连接集合
        self.upstream = []#上游连接集合
        self.output = 0 #输出
        self.delta = 0 #

    def set_output(self,output):
        #输入层的输出函数
        self.output = output

    def append_downstream_connection(self,conn):
        '''
        添加一个到下游节点的连接
        '''
        self.downstream.append(conn)#downstream是一个列表，使用append方法将conn添加进去

    def append_upstream_connection(self,conn):
        '''
        添加一个到上游节点的连接
        '''
        self.upstream.append(conn)#upstream是一个列表，使用append方法将conn添加进去

    def calc_output(self):
        output = reduce(lambda ret, conn: ret + conn.upstream_node.output * conn.weight, self.upstream, 0)#reduce函数对上游连接的输出进行求和
        self.output = sigmoid(output)#利用sigmoid进行激活

    def calc_hidden_layer_delta(self):
        # 隐含层求梯度
        downstream_delta = reduce(
            lambda ret, conn: ret + conn.downstream_node.delta * conn.weight,
            self.downstream, 0.0)
        self.delta = self.output * (1 - self.output) * downstream_delta#利用推导出的公式进行计算梯度

    def calc_output_layer_delta(self, label):
        # 输出层求梯度
        self.delta = self.output * (1 - self.output) * (label - self.output)

    def __str__(self):
        # 打印节点信息
        node_str = '%u-%u: output: %f delta: %f' % (self.layer_index, self.node_index, self.output, self.delta)
        downstream_str = reduce(lambda ret, conn: ret + '\n\t' + str(conn), self.downstream, '')
        upstream_str = reduce(lambda ret, conn: ret + '\n\t' + str(conn), self.upstream, '')
        return node_str + '\n\tdownstream:' + downstream_str + '\n\tupstream:' + upstream_str
        # %u：表示无符号的十进制数 \t表示4个空格


# constNode对象，为了实现一个输出恒为1的节点（计算偏置b时需要）
class ConstNode(object):
    def __init__(self,layer_index,node_index):
        self.layer_index = layer_index
        self.node_index = node_index
        self.downstream = []
        self.output = 1

    def append_downstream_connection(self, conn):
        self.downstream.append(conn)

    def calc_hidden_layer_delta(self):
        downstream_delta = reduce(
            lambda ret, conn: ret + conn.downstream_node.delta * conn.weight,
            self.downstream, 0.0)
        self.delta = self.output * (1 - self.output) * downstream_delta

    def __str__(self):
        node_str = '%u-%u: output: 1' % (self.layer_index, self.node_index)
        downstream_str = reduce(lambda ret, conn: ret + '\n\t' + str(conn), self.downstream, '')
        return node_str + '\n\tdownstream:' + downstream_str


# Layer对象，负责初始化一层。此外，作为Node的集合对象，提供对Node集合的操作。
class Layer(object):
    def __init__(self, layer_index, node_count):
        self.layer_index = layer_index
        self.nodes = []
        for i in range(node_count):
            self.nodes.append(Node(layer_index, i))#将每个节点初始化，为每个神经元标记层号以及序号
        self.nodes.append(ConstNode(layer_index, node_count))#添加值恒为1的节点

    def set_output(self, data):
        data = list(data)
        for i in range(len(data)):
            self.nodes[i].set_output(data[i])#对每个节点进行数值初始化

    def calc_output(self):
        for node in self.nodes[:-1]:
            node.calc_output()#对每个节点计算输出值

    def dump(self):
        for node in self.nodes:
            print(node)

# Connection对象，主要职责是记录连接的权重，以及这个连接所关联的上下游节点。
class Connection(object):
    def __init__(self, upstream_node, downstream_node):
        self.upstream_node = upstream_node
        self.downstream_node = downstream_node
        self.weight = random.uniform(-0.1, 0.1)#权值初始化为-0.1到0.1的一个值
        self.gradient = 0.0

    def calc_gradient(self):
        self.gradient = self.downstream_node.delta * self.upstream_node.output  #计算梯度

    def update_weight(self, rate):
        self.calc_gradient()
        self.weight += rate * self.gradient #更新权值

    def get_gradient(self):
        return self.gradient

    def __str__(self):
        return '(%u-%u) -> (%u-%u) = %f' % (
            self.upstream_node.layer_index,
            self.upstream_node.node_index,
            self.downstream_node.layer_index,
            self.downstream_node.node_index,
            self.weight)



# Connections对象，提供Connection集合操作。
class Connections(object):
    def __init__(self):
        self.connections = []

    def add_connection(self, connection):
        self.connections.append(connection)

    def dump(self):
        for conn in self.connections:
            print(conn)

# Network对象，提供API。
class Network(object):
    def __init__(self, layers):
        '''
        初始化一个全连接神经网络
        layers:存放层节点
        '''
        self.connections = Connections()
        self.layers = []  # 数组，每个元素存放每层结点数
        layer_count = len(layers)  # 表示层数
        # node_count = 0 # 节点总数
        # 初始化层数
        for i in range(layer_count):  # 0到(层数-1)
            self.layers.append(Layer(i, layers[i]))
        # 初始化连接边
        for layer in range(layer_count - 1):  # 表示最后一层不包括进去
            connections = [Connection(upstream_node, downstream_node)
                           for upstream_node in self.layers[layer].nodes
                           for downstream_node in self.layers[layer + 1].nodes[:-1]]

            for conn in connections:
                self.connections.add_connection(conn)
                conn.upstream_node.append_upstream_connection(conn)
                conn.downstream_node.append_downstream_connection(conn)

    def train(self, labels, data_set, rate, iteration):
        '''
        训练神经网络
        label: 数组，训练的样本标签，每个元素是一个样本的标签
        data_set: 二维数组，训练样本特征
        '''
        for i in range(iteration):
            for d in range(len(data_set)):
                self.train_one_sample(labels[d], data_set[d], rate)

    def train_one_sample(self, label, sample, rate):
        '''内部函数，用一个样本训练网络'''
        self.predict(sample)
        self.calc_delta(label)
        self.update_weight(rate)

    def calc_delta(self, label):
        '''内部函数，计算每个节点的delta'''
        output_nodes = self.layers[-1].nodes  # [-1]表示最后一层数据
        label2 = list(label)
        for i in range(len(label2)):
            output_nodes[i].calc_output_layer_delta(label2[i])
        for layer in self.layers[-2::-1]:  # [-2::-1]表示 将layers列表反过来，然后除去之前的最后一元素
            for node in layer.nodes:
                node.calc_hidden_layer_delta()

    def update_weight(self, rate):
        '''内部函数，更新每个连接权重'''
        for layer in self.layers[:-1]:
            for node in layer.nodes:
                for conn in node.downstream:
                    conn.update_weight(rate)

    def calc_gradient(self):
        '''计算每个连接的梯度'''
        for layer in self.layers[:-1]:
            for node in layer.nodes:
                for conn in node.downstream:
                    conn.calc_gradient()

    def get_gradient(self, sample, label):
        '''
        获得网络在一个样本下，每个连接上的梯度
        label：样本标签
        sample：样本输入
        '''
        self.predict(sample)
        self.calc_delta(label)
        self.calc_gradient()

    def predict(self, sample):
        '''
        根据输入的样本预测输出值
        sanple：样本特征，网络输入向量
        '''
        self.layers[0].set_output(sample)
        for i in range(1, len(self.layers)):
            self.layers[i].calc_output()
        return map(lambda node: node.output, self.layers[-1].nodes[:-1])  # [-1]最后一个元素，[:-1]除最后一个元素的所有元素

    def dump(self):
        '''打印网络信息'''
        for layer in self.layers:
            layer.dump()


class Normalizer(object):
    def __init__(self):
        self.mask = [
            0x1, 0x2, 0x4, 0x8, 0x10, 0x20, 0x40, 0x80
        ]

    def norm(self, number):
        return map(lambda m: 0.9 if number & m else 0.1, self.mask)

    def denorm(self, vec):
        binary = list(map(lambda i: 1 if i > 0.5 else 0, vec))
        for i in range(len(self.mask)):
            binary[i] = binary[i] * self.mask[i]
        return reduce(lambda x, y: x + y, binary)


def mean_square_error(vec1, vec2):
    return 0.5 * reduce(lambda a, b: a + b,
                        map(lambda v: (v[0] - v[1]) * (v[0] - v[1]),
                            zip(vec1, vec2)
                            )
                        )


def gradient_check(network, sample_feature, sample_label):
    '''
    梯度检查
    network: 神经网络对象
    sample_feature: 样本的特征
    sample_label: 样本的标签
    '''
    # 计算网络误差
    network_error = lambda vec1, vec2: \
        0.5 * reduce(lambda a, b: a + b,
                     map(lambda v: (v[0] - v[1]) * (v[0] - v[1]),
                         zip(vec1, vec2)))

    # 获取网络在当前样本下每个连接的梯度
    network.get_gradient(sample_feature, sample_label)

    # 对每个权重做梯度检查
    for conn in network.connections.connections:
        # 获取指定连接的梯度
        actual_gradient = conn.get_gradient()

        # 增加一个很小的值，计算网络的误差
        epsilon = 0.0001
        conn.weight += epsilon
        error1 = network_error(network.predict(sample_feature), sample_label)

        # 减去一个很小的值，计算网络的误差
        conn.weight -= 2 * epsilon  # 刚才加过了一次，因此这里需要减去2倍
        error2 = network_error(network.predict(sample_feature), sample_label)

        # 根据式6计算期望的梯度值
        expected_gradient = (error2 - error1) / (2 * epsilon)

        # 打印
        print('expected gradient: \t%f\nactual gradient: \t%f' % (
            expected_gradient, actual_gradient))


def train_data_set():
    normalizer = Normalizer()
    data_set = []
    labels = []
    for i in range(0, 256, 8): # 从0到256，不包括256，步长为8，每8个为一个间隔
        n = normalizer.norm(int(random.uniform(0, 256)))
        data_set.append(n)
        labels.append(n)
    return labels, data_set


def train(network):
    labels, data_set = train_data_set()
    network.train(labels, data_set, 0.3, 50)


def test(network, data):
    normalizer = Normalizer()
    norm_data = normalizer.norm(data)
    predict_data = network.predict(norm_data)
    print
    ('\ttestdata(%u)\tpredict(%u)' % (
        data, normalizer.denorm(predict_data)))


def correct_ratio(network):
    normalizer = Normalizer()
    correct = 0.0;
    for i in range(256):
        if normalizer.denorm(network.predict(normalizer.norm(i))) == i:
            correct += 1.0
    print('correct_ratio: %.2f%%' % (correct / 256 * 100))


def gradient_check_test():
    net = Network([2, 2, 2])
    sample_feature = [0.9, 0.1]
    sample_label = [0.9, 0.1]
    gradient_check(net, sample_feature, sample_label)


if __name__ == '__main__':
    net = Network([8, 3, 8])
    train(net)
    net.dump()
    correct_ratio(net)