#!/usr/bin/python
# -*- coding: utf-8 -*-
#coding=utf-8
import numpy as np
import xlwt


def establish_matrix(data, m_dim, l_max, average):
    dim = m_dim * l_max
    num_data = data.shape[1]
    amount = 0
    for i in range(0, num_data):
        amount = amount + data[0, i]
    average = 1./num_data * amount
    print(average)
    zero_data = []
    j = 0
    for i in range(0, num_data - 1):
        if (data[0, i] - average) * (data[0, i + 1] - average) <= 0:
            zero_data.append(i)
            j = j + 1
    num_zero_data = j
    w = np.zeros((1, m_dim * dim + 1))
    for i in range(1, m_dim + 1):
        j = 0
        l = []
        start_point = (i-1)*dim
        length = i * l_max
        count = 0
        while j + i <= num_zero_data-1:
            l.append(zero_data[j + i] - zero_data[j] - 1)
            j = j + 1
        # print(L)
        for j in range(1, length + 1):
            for t in range(0, len(l)):
                if l[t] == j:
                    w[0, start_point + j] = w[0, start_point + j] +1
                    count = count + 1

        if i == 1:
            for t in range(0, len(l)):
                if l[t] == 0:
                    w[0, 0] = w[0, 0] + 1
            count = count + w[0, 0]
            if count != 0:
                w[0, 0:length + 1] = 1./count * w[0, 0:length + 1]
        else:
            if count != 0:
                w[0, start_point + 1:start_point + length + 1] = 1./count * w[0, start_point + 1:start_point + length + 1]
    return w
"""
建立声纹特征矩阵
输入采样数据矩阵、期望的维数、相邻过零点间最大采样点数
输出声纹特征矩阵
"""


def distance(w1, w2):
    dis = np.linalg.norm(w1 - w2)
    return dis
"""求行向量的欧氏距离
"""


def model_matrix(model_data, m_dim, l_max, average):
    row = model_data.shape[0]
    dim = m_dim * l_max
    model = np.zeros((row, m_dim * dim + 1))
    for i in range(0, row):
        data = model_data[i, :]
        model[i, :] = establish_matrix(data, m_dim, l_max, average)
    return model
"""求解模板的声纹特征矩阵
"""


def comparison( model, sample_data, m_dim, l_max, average):
    num_model = w_model.shape[0]
    dim = m_dim * l_max
    d = np.zeros((num_model, m_dim))
    d_min = []
    w2 = establish_matrix(sample_data, m_dim, l_max, average)
    for i in range(0, num_model):
        w1 = np.matrix(model[i, :])
        for j in range(1, m_dim + 1):
            y1 = np.zeros((1, j * dim + 1))
            y2 = np.zeros((1, j * dim + 1))
            y1 = w1[0, 0: j * dim + 1]
            y2 = w2[0, 0: j * dim + 1]
            d[i, j - 1] = distance(y1, y2)
    for j in range(0, m_dim):
        delta = []
        for i in range(0, num_model):
            delta.append(d[i, j])
        d_min.append(min(delta)*1000)
    return d_min
    # if min_distance <= threshold:
    #     print("Success identification!")
    # else:
    #     print("Failed identification!")
"""对比识别声纹
"""

data_file = xlwt.Workbook()
sheet1 = data_file.add_sheet(u'sheet1',cell_overwrite_ok = True)
data_file.save('Excel_sheet1.xls')
sheet1.write(0,0,'dimension')
sheet1.write(0,1,'tank')
sheet1.write(0,2,'plane')
sheet1.write(0,3,'train')
sheet1.write(0,4,'human')
sheet1.write(1,0,'1D')
sheet1.write(2,0,'2D')
sheet1.write(3,0,'3D')
sheet1.write(4,0,'4D')
"""建立Excel表
"""

data_of_tank = np.matrix(np.loadtxt("tank.txt", unpack='true'))
model_data = np.matrix(np.loadtxt("human model.txt", unpack='true'))
w_model = model_matrix(model_data, 4, 10, 4096)
result1 = comparison( w_model, data_of_tank, 4, 10, 4096)
for i in range(0, 4):
    sheet1.write(i+1, 1, result1[i])


data_of_plane = np.matrix(np.loadtxt("plane.txt", unpack='true'))
# model_data_of_plane = np.matrix(np.loadtxt("plane model.txt", unpack='true'))
result2 = comparison( w_model, data_of_plane, 4, 10, 4096)
for i in range(0, 4):
    sheet1.write(i+1, 2, result2[i])


data_of_train = np.matrix(np.loadtxt("train.txt", unpack='true'))
# model_data_of_train = np.matrix(np.loadtxt("train model.txt", unpack='true'))
result3 = comparison( w_model, data_of_train, 4, 10, 4096)
for i in range(0, 4):
    sheet1.write(i+1, 3, result3[i])


data_of_human = np.matrix(np.loadtxt("human.txt", unpack='true'))
# model_data_of_human = np.matrix(np.loadtxt("human model.txt", unpack='true'))
result4 = comparison( w_model, data_of_human, 4, 10, 4096)
for i in range(0, 4):
    sheet1.write(i+1, 4, result4[i])
"""将计算结果写入Excel表
"""
data_file.save('Excel_sheet1.xls')


