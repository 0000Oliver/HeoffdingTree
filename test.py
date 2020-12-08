# from sklearn import tree
# from sklearn.datasets import load_iris
# from sklearn.ensemble import GradientBoostingClassifier
#
# import os
# from PIL import Image
# import cv2
# import numpy as np
# import random
#
#
# def read_pics(file_dir):
#     pos_root = file_dir+"/pos"
#     neg_root = file_dir+"/neg"
#     pos_dirlist = os.listdir(pos_root)
#     neg_dirlist = os.listdir(neg_root)
#
#
#     X = []
#     for im_dir in pos_dirlist:
#         # im = cv2.imread(os.path.join(pos_root,im_dir))
#         im = Image.open(os.path.join(pos_root,im_dir))
#         im_array = np.array(im)
#         im_array = list(im_array.flatten())
#         im_array.append(1)
#         X.append(im_array)
#     for im_dir in neg_dirlist:
#         im = Image.open(os.path.join(neg_root,im_dir))
#         im_array = np.array(im)
#         im_array = list(im_array.flatten())
#         im_array.append(0)
#         X.append(im_array)
#     random.shuffle(X)
#     X = np.array(X)
#     y = X[:,-1]
#     X = X[:,:-1]
#     print( X)
#     print(y)
#     return X,y
# train_root = "/Users/wangqiang/Source/facedata/train"
# test_root = "/Users/wangqiang/Source/facedata/test"
# X_train,y_train = read_pics(train_root)
# X_test,y_test = read_pics(test_root)
# # clf = GradientBoostingClassifier(random_state=10)
# clf = tree.DecisionTreeClassifier()
# clf = clf.fit(X_train, y_train)
# score = clf.score(X_test, y_test)
# print(score)


# coding=utf8

import shutil
import os
source = "/Users/wangqiang/Source"  # "/home/wangqiang/Source"
train_root = source + "/facedata/train"
pos_dirlist = os.listdir(train_root+"/pos")
neg_dirlist = os.listdir(train_root+"/neg")
for dir in neg_dirlist:
    oldpath = os.path.join(train_root+"/neg",dir)
    newpath = os.path.join("/Users/wangqiang/Source/facedata/mixed_train",dir[:-4]+"n.bmp")
    print(oldpath)
    print(newpath)
    shutil.copy(oldpath,newpath)

#
# # 打开存储文件命名规则的文件
# data = xlrd.open_workbook('C:ccc\\新新编号.xls')
# # 打开工作表
# table = data.sheet_by_name(u'Sheet1')
# # 获取第一列所有内容，返回的是数组
# name = table.col_values(0)
# # 获取第二列所有内容，返回的是数组
# bank = table.col_values(1)
# # 获取行数，返回的是int
# nrows = table.nrows
# for i in range(nrows):
#         bank1 = bank[i]
#         # 这里上下两行的代码可忽略，因为我是想把返回的数组里的每个先赋值变量bank1，再截取字符串的前4个
#         bank2 = bank1[0:4]
#         # 循环一次复制一个文件，文件名由变量组成
#         shutil.copy("C:\\ccc\\新新人类模板.xlsx",
#                     "C:\\ccc\\"+'新新-'+name[i]+'-'+bank2+'-.xlsx')
