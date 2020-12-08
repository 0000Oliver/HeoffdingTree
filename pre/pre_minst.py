from hoeffdingtree import *
import random
def convert(imgf, labelf, outf, n):
    f = open(imgf, "rb")
    o = open(outf, "w")
    l = open(labelf, "rb")

    f.read(16)
    l.read(8)
    images = []

    for i in range(n):
        image = [ord(l.read(1))]
        for j in range(28 * 28):
            image.append(ord(f.read(1)))
        images.append(image)

    for image in images:
        o.write(",".join(str(pix) for pix in image) + "\n")
    f.close()
    o.close()
    l.close()


# convert("/Users/wangqiang/Source/minst/train-images-idx3-ubyte", "/Users/wangqiang/Source/minst/train-labels-idx1-ubyte",
#         "/Users/wangqiang/Source/minst/mnist_train.csv", 60000)
# convert("/Users/wangqiang/Source/minst/t10k-images-idx3-ubyte", "/Users/wangqiang/Source/minst/t10k-labels-idx1-ubyte",
#         "/Users/wangqiang/Source/minst/mnist_test.csv", 10000)

import csv
import matplotlib.pyplot as plt
import numpy as np

from PIL import Image
testdata = csv.reader(open("/Users/wangqiang/Source/minst/mnist_test.csv",'r'))
traindata = csv.reader(open("/Users/wangqiang/Source/minst/mnist_train.csv",'r'))
for data in traindata:
    if data[0] =="0":

        im_array = np.array(data[1:]).reshape(28,28)
        im_array = np.mat( np.uint8(im_array))

        img = Image.fromarray(im_array)
        img.resize((200,200),Image.ANTIALIAS)
        print(img)
        img.show()
        img.save("../source/0.jpg")
        break



def read_minst_csv(filepath,name):
    data = csv.reader(open(filepath, 'r'))
    headers = list(data[0])
    headers[0] = "label"
    attributes = []
    # 第一位是label，是字符型
    labels = []
    for i in range(10):
        labels.append(str(i))
    attributes.append(Attribute(str(headers[0]),labels , 'Nominal'))
    # 图片压缩为一维数组作为所有属性，所有的都是数字型
    for i in range(1,len(headers)):
        attributes.append(Attribute(str(headers[i]), att_type='Numeric'))

    class_index = 0
    instances = []
    for d in data:
        instances.append(d)

    random.shuffle(instances)
    dataset = Dataset(attributes, class_index, name=name)
    for inst in instances:
        inst[0] = int(attributes[0].index_of_value(str(inst[0])))  # 只有最后一个label是字符型的
        dataset.add(Instance(att_values=inst))
    return dataset
