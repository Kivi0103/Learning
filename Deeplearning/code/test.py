import numpy as np

# from functools import reduce
# lst = [1, 2, 3, 4, 5]
# a = [2, 2, 7, 4, 5]
# # result = reduce(lambda x, y: x+y, lst, 1)
# print(zip(list, a))

# # print(result)   # 输出 120，即 1*2*3*4*5
# import random
#
#
# def square(x):
#     return x ** 2
#
#
# x = [1, 2, 3, 4]
# y = map(square, x)
# print(y)
# # 输出 [1, 4, 9, 16]
#
# for i in range(28):
#     print(i)





# def get_one_sample(picture):
#     """
#     内部函数，将图像转化为样本的输入向量
#     """
#     sample = []
#     for i in range(28):  # ????感觉没什么作用呀，为什么不直接用上面的picture，或者直接在picture上就改成这样
#         for j in range(28):
#             sample.append(picture[i][j])
#     return sample
#
# picture = []
# for i in range(28):  #将28行像素加入picture列表
#     picture.append([])
#     for j in range(28): #每一行28个像素值
#         picture[i].append(random.randint(1,10))
# print(picture)
#
# print(get_one_sample(picture))


# mask = [
#             0x1, 0x2, 0x4, 0x8, 0x10, 0x20, 0x40, 0x80
#         ]
# data_set = []
# for i in range(0,10):
#     data = list(map(lambda m: 0.9 if i & m else 0.1, mask))
#     data_set.append(data)
#
# def transpose(args):
#     return list(list(map(lambda line: np.array(line).reshape(len(line), 1), args)))
# print(data_set)
# print(transpose(data_set))

a = np.array([0.1,0.1,0.1,0.1,0.1,0.9,0.1,0.1,0.1,0.1]).reshape(10,1)
print(a.shape)
b = np.array([[1,2,3],[4,5,6],[7,8,9],[0,1,2],[3,4,5],
              [1,2,3],[4,5,6],[7,8,9],[0,1,2],[3,4,5]])
print(b.shape)
c = a-b
print(c)