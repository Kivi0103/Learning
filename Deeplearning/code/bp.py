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
        layer_index: 节点所属的层的编号
        node_index: 节点的编号
        '''
        self.layer_index = layer_index
        self.node_index = node_index
        self.downstream = []
        self.upstream = []
        self.output = 0
        self.delta = 0

    def set_output(self,output):
        '''
         设置节点的输出值。如果节点属于输入层会用到这个函数。
         '''
        self.output = output

    def append_downstream_connection(self,conn):
        '''
        添加一个到下游节点的连接
        '''
        self.downstream.append(conn)

    def append_upstream_connection(self,conn):
        '''
        添加一个到上游节点的连接
        '''
        self.upstream.append(conn)

    def calc_output(self):
        output = reduce(lambda ret, conn: ret + conn.upstream_node.output * conn.weight, self.upstream, 0)
        self.output = sigmoid(output)

    def calc_hidden_layer_delta(self):
        # 隐含层求梯度
        downstream_delta = reduce(
            lambda ret, conn: ret + conn.downstream_node.delta * conn.weight,
            self.downstream, 0.0)
        self.delta = self.output * (1 - self.output) * downstream_delta

    def calc_output_layer_delta(self, label):
        # 输出层求梯度
        self.delta = self.output * (1 - self.output) * (label - self.output)

    def __str__(self):
        # 打印节点信息
        node_str = '%u-%u: output: %f delta: %f' % (self.layer_index, self.node_index, self.output, self.delta)
        downstream_str = reduce(lambda ret, conn: ret + '\n\t' + str(conn), self.downstream, '')
        upstream_str = reduce(lambda ret, conn: ret + '\n\t' + str(conn), self.upstream, '')
        return node_str + '\n\tdownstream:' + downstream_str + '\n\tupstream:' + upstream_str