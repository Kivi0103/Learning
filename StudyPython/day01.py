# 整除符号//，向下取整
print(11//2)

"""这是三引号注释"""
str1 = """这也是字符串"""
print(str1)
#结果为：这也是字符串


# 任何类型都可以转换成字符串
print(str(object),str(11),str(12.23),str([2,3,4,5]),str({12,3,4,5}),str({123:3,"23":323}))
#结果为：<class 'object'> 11 12.23 [2, 3, 4, 5] {4, 3, 12, 5} {123: 3, '23': 323}

# 浮点数转换成小数会丢失精度
print(int(12.33))  #结果为：12


# 列表（List）	有序的可变序列	Python中使用最频繁的数据类型，可有序记录一堆数据
# 元组（Tuple）	有序的不可变序列	可有序记录一堆不可变的Python数据集合


# 字符串不能与数值进行拼接
# print("12345"+12345)
# 上述代码会报错：can only concatenate str (not "int") to str


# 字符串格式化  %d表示整数占位符，%f表示浮点数占位符 %s表示字符串占位符
# 格式化的格式："占位符和要输出的字符串"%(变量1，变量2.。。。。)
a = 267
b = 2.67
s = "我爱python"
print("%d测试%s测试%f"%(a,s,b))#结果为：267测试我爱python测试2.670000

#字符串格式化中的精度控制
print("测试%7.2f"%(12.434)) #%7.2f表示控制该浮点数的宽度为7，小数点也占一位，小数部分为2，小数部分四舍五入，不足的宽度由空格补足
# 结果为：测试  12.43
print("测试%.2f"%(12.434)) #不限宽度，只限小数位数，小数四舍五入
#结果为：测试12.43


#字符格式化 快速方法
print(f"测试{object}测试{23}")#使用f加{}的方法，不做精度控制，直接打印
