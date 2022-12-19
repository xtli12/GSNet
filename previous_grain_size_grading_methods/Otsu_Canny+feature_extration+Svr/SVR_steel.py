# import numpy as np
#
# raw_data = np.loadtxt('./canny_output/canny_result.txt')  # 读取数据文件
# X = raw_data[:,:,-1 ]  # 分割自变量
# y = raw_data[:, -1]
import torch
from sklearn.svm import SVR
from sklearn.model_selection import cross_val_score
import codecs
import matplotlib.pyplot as plt
import numpy as np



f = codecs.open('./otsu_output/otsu_train_result.txt', mode='r', encoding='utf-8')  # 打开txt文件，以‘utf-8'编码读取



line = f.readline()  # 以行的形式进行读取文件 

x_train = []
y_train = []

while line:

    a = line.split()

    b = a[0:1]  # 这是选取需要读取的位数
    # b[0] = b[0] / 10  #数据溢出。除10来减小
    c = a[1:2]

    x_train.append(b)  # 将其添加在列表之中
    y_train.append(c)  # 将其添加在列表之中
    line = f.readline()

f.close()

f = codecs.open('./otsu_valid_output/otsu_val_result.txt', mode='r', encoding='utf-8')  # 打开txt文件，以‘utf-8'编码读取


line = f.readline()  # 以行的形式进行读取文件

x_test = []
y_test = []
y_test_pred =[]
while line:

    a = line.split()

    b = a[0:1]  # 这是选取需要读取的位数
    # b = a[0:1]/10
    c = a[1:2]

    x_test.append(b)  # 将其添加在列表之中
    y_test.append(c)  # 将其添加在列表之中
    line = f.readline()

f.close()






# for i in list2:
#
#     print(i)
Svr = SVR(kernel='rbf',C=1,gamma=0.5)
Svr.fit(x_train, y_train)
y_pred = Svr.predict(x_test)
y_pred=y_pred.tolist()


ac = 0
acc0 =0
acc5 =0
idx_to_class = {10: '10.0', 10.5: '10.5', 11.0: '11.0', 11.5: '11.5', 12.0: '12.0', 12.5: '12.5', 13.0: '13.0', 6.5: '6.5', 7.0: '7.0',
                7.5: '7.5', 8.0: '8.0', 8.5: '8.5', 9.0: '9.0', 9.5: '9.5'}
for i,pred in enumerate(y_pred):
    # v = y_pred[i]
    l = len(y_pred)
    b = y_test[i]
    b = b[0]
    target = [k for k, v in idx_to_class.items() if v == b]  #以值找键，将字符串转换为浮点数
    m = target [0]
    if abs(m - pred)==0:
        acc0 = 1
    elif abs(m - pred) < 0.5:
        acc5 = 1
    ac += (0.4 * acc0 + 0.6 * acc5) / l
# ac = (0.4 * acc0 + 0.6 * acc5) / l
print(ac)
