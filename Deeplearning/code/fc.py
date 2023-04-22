from datetime import datetime
from functools import reduce

import numpy as np
import random
from activators import SigmoidActivator,IdentityActivator

# 全连接层实现类
class FullConnectedLayer(object):
    def __init__(self, input_size, output_size,
                 activator):
        '''
        构造函数
        input_size: 本层输入向量的维度
        output_size: 本层输出向量的维度
        activator: 激活函数
        '''
        self.input_size = input_size
        self.output_size = output_size
        self.activator = activator
        # 权重数组W
        self.W = np.random.uniform(-0.1, 0.1,
            (output_size, input_size))
        # 偏置项b
        self.b = np.zeros((output_size, 1))
        # 输出向量
        self.output = np.zeros((output_size, 1))

    def forward(self, input_array):
        '''
        前向计算
        input_array: 输入向量，维度必须等于input_size
        '''
        # 式2
        self.input = input_array
        self.output = self.activator.forward(
            np.dot(self.W, input_array) + self.b)

    def backward(self, delta_array):
        '''
        反向计算W和b的梯度
        delta_array: 从上一层传递过来的误差项
        '''
        # 式8
        self.delta = self.activator.backward(self.input) * np.dot(
            self.W.T, delta_array)
        self.W_grad = np.dot(delta_array, self.input.T)
        self.b_grad = delta_array

    def update(self, learning_rate):
        '''
        使用梯度下降算法更新权重
        '''
        self.W += learning_rate * self.W_grad
        self.b += learning_rate * self.b_grad

    def dump(self):
        print ('W: %s\nb:%s' % (self.W, self.b))


# 神经网络类
class Network(object):

    #layers中表示各层节点个数，[748,300,10]
    def __init__(self, layers):
        '''
        构造函数
        '''
        self.layers = []
        for i in range(len(layers) - 1):
            self.layers.append(
                FullConnectedLayer(
                    layers[i], layers[i+1],
                    SigmoidActivator()
                )
            )
            print(f"Layer {i}: input shape {layers[i]}, output shape {layers[i + 1]}")

    def predict(self, sample):
        '''
        使用神经网络实现预测
        sample: 输入样本
        '''
        output = sample
        for layer in self.layers:
            layer.forward(output)
            output = layer.output
        return output

    def train(self, labels, data_set, rate, epoch):
        '''
        训练函数
        labels: 样本标签
        data_set: 输入样本
        rate: 学习速率
        epoch: 训练轮数
        '''
        for i in range(epoch):
            for d in range(len(data_set)):
                self.train_one_sample(labels[d], data_set[d], rate)


    def train_one_sample(self, label, sample, rate):
        self.predict(sample)
        self.calc_gradient(label)
        self.update_weight(rate)

    def calc_gradient(self, label):
        # print(self.layers[-1].output.shape)
        # print(self.layers[-1].activator.backward(self.layers[-1].output).shape)
        # print(np.label.shape())
        # print(self.layers[-1].output.shape)
        a = np.array(self.layers[-1].activator.backward(self.layers[-1].output))
        # print(a.shape)
        # print(np.array(label).reshape(10,1).shape)
        # print(np.array(self.layers[-1].output).shape)
        b = np.array(label).reshape(10,1) - np.array(self.layers[-1].output)
        # print(a.shape)
        # print(b.shape)
        delta = a*b
        print(delta.shape)
        for layer in self.layers[::-1]:
            layer.backward(delta)
            delta = layer.delta
        return delta

    def update_weight(self, rate):
        for layer in self.layers:
            layer.update(rate)

    def dump(self):
        for layer in self.layers:
            layer.dump()

    def loss(self, output, label):
        return 0.5 * ((label - output) * (label - output)).sum()

    def gradient_check(self, sample_feature, sample_label):
        '''
        梯度检查
        network: 神经网络对象
        sample_feature: 样本的特征
        sample_label: 样本的标签
        '''

        # 获取网络在当前样本下每个连接的梯度
        self.predict(sample_feature)
        self.calc_gradient(sample_label)

        # 检查梯度
        epsilon = 10e-4
        for fc in self.layers:
            for i in range(fc.W.shape[0]):
                for j in range(fc.W.shape[1]):
                    fc.W[i,j] += epsilon
                    output = self.predict(sample_feature)
                    err1 = self.loss(sample_label, output)
                    fc.W[i,j] -= 2*epsilon
                    output = self.predict(sample_feature)
                    err2 = self.loss(sample_label, output)
                    expect_grad = (err1 - err2) / (2 * epsilon)
                    fc.W[i,j] += epsilon
                    print ('weights(%d,%d): expected - actural %.4e - %.4e' % (
                        i, j, expect_grad, fc.W_grad[i,j]))

from minist import get_training_data_set
from minist import get_test_data_set
from minist import evaluate

def train_and_evaluate():
    last_error_ration = 1.0
    epoch = 0
    train_data_set,train_labels = get_training_data_set()
    test_data_set, test_labels = get_test_data_set()
    network = Network([784,300,10])
    while True:
        epoch += 1
        network.train(train_labels,train_data_set,0.3,10)
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
