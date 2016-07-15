import numpy as np

# f = open("111.txt",'r')
# a = f.read()
# b=a.split(",")
# y=np.loadtxt("111.txt", unpack='true')
# y = np.matrix(y)
# print(y)
m = np.matrix([1,2,3,4])
# d = []
# num = 4
# amount = 0
# for i in range(0, num):
#     amount = amount + m[i]
# average = 1. / num * amount
# print(average)
# print(m[1:3])
m = m[0,0:4]
print(m[0,0:4])
