# from functools import reduce
# lst = [1, 2, 3, 4, 5]
# a = [2, 2, 7, 4, 5]
# # result = reduce(lambda x, y: x+y, lst, 1)
# print(zip(list, a))

# # print(result)   # 输出 120，即 1*2*3*4*5


def square(x):
    return x ** 2


x = [1, 2, 3, 4]
y = map(square, x)
print(y)
# 输出 [1, 4, 9, 16]
