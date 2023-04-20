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





def get_one_sample(picture):
    """
    内部函数，将图像转化为样本的输入向量
    """
    sample = []
    for i in range(28):  # ????感觉没什么作用呀，为什么不直接用上面的picture，或者直接在picture上就改成这样
        for j in range(28):
            sample.append(picture[i][j])
    return sample

picture = []
for i in range(28):  #将28行像素加入picture列表
    picture.append([])
    for j in range(28): #每一行28个像素值
        picture[i].append(random.randint(1,10))
print(picture)

print(get_one_sample(picture))
