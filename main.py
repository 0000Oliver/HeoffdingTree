import csv
from hoeffdingtree import *
import os
from PIL import Image
import numpy as np

import random
def open_dataset(filename, class_index, probe_instances=100):
    """ Open and initialize a dataset in CSV format.
    The CSV file needs to have a header row, from where the attribute names will be read, and a set
    of instances containing at least one example of each value of all nominal attributes.

    Args:
        filename (str): The name of the dataset file (including filepath).
        class_index (int): The index of the attribute to be set as class.
        probe_instances (int): The number of instances to be used to initialize the nominal 
            attributes. (default 100)

    Returns:
        Dataset: A dataset initialized with the attributes and instances of the given CSV file.
    """
    if not filename.endswith('.csv'):
        raise TypeError(
            'Unable to open \'{0}\'. Only datasets in CSV format are supported.'
            .format(filename))
    with open(filename) as f:
        fr = csv.reader(f)
        headers = next(fr)

        att_values = [[] for i in range(len(headers))]
        instances = []
        try:
            for i in range(probe_instances):
                inst = next(fr)
                instances.append(inst)
                for j in range(len(headers)):
                    try:
                        inst[j] = float(inst[j])
                        att_values[j] = None
                    except ValueError:
                        inst[j] = str(inst[j])
                    if isinstance(inst[j], str):
                        if att_values[j] is not None:
                            if inst[j] not in att_values[j]:
                                att_values[j].append(inst[j])
                        else:
                            raise ValueError(
                                'Attribute {0} has both Numeric and Nominal values.'
                                .format(headers[j]))
        # Tried to probe more instances than there are in the dataset file
        except StopIteration:
            pass

    attributes = []
    for i in range(len(headers)):
        if att_values[i] is None:
            attributes.append(Attribute(str(headers[i]), att_type='Numeric'))
        else:
            attributes.append(Attribute(str(headers[i]), att_values[i], 'Nominal'))

    dataset = Dataset(attributes, class_index)
    for inst in instances:
        for i in range(len(headers)):
            if attributes[i].type() == 'Nominal':
                inst[i] = int(attributes[i].index_of_value(str(inst[i])))
        dataset.add(Instance(att_values=inst))
    
    return dataset

def read_dataset(name_file,data_file,class_index=-1,probe_instances=None,name ="New dataset"):
    '''
    Open and initialize a dataset in file format.
    按照文件的格式初始化UCI数据库的dataset类
    :param name_file:
    :param data_file:
    :param class_index:
    :param probe_instances:
    :return:
    '''
    with open(name_file) as f:
        lines = f.readlines()

        headers = []

        att_values=[]
        for line in lines[1:]:
            if line=='\n':
                continue
            line = line.replace(' ','')
            headers.append(line.split(':')[0])
            att_value = line.split(':')[1].strip()[:-1].split(',')
            if len(att_value)==1:
                att_value = None
            att_values.append(att_value)
        headers.append('label')
        att_values.append(lines[0].replace(' ', '').strip()[:-1].split(','))
    class_index = len(headers)-1
    with open(data_file) as f:
        lines =f.readlines()
        instances = []
        if probe_instances :
            count = probe_instances
        else:
            count = len(lines)-1
        for i in range(count):
            line = lines[i].replace(' ', '').strip()

            if line=='':
                continue
            inst = line.split(',')

            for j in range(len(headers)-1):
                try:
                    inst[j] = float(inst[j])
                except ValueError:
                    inst[j] = str(inst[j])
            instances.append(inst)
    attributes = []
    for i in range(len(headers)):
        if att_values[i] is None:
            attributes.append(Attribute(str(headers[i]), att_type='Numeric'))
        else:
            attributes.append(Attribute(str(headers[i]), att_values[i], 'Nominal'))
    dataset = Dataset(attributes, class_index,name=name)
    for inst in instances:
        for i in range(len(headers)):
            if attributes[i].type() == 'Nominal':
                inst[i] = int(attributes[i].index_of_value(str(inst[i])))
        dataset.add(Instance(att_values=inst))

    return dataset
def choic_dataset(dataname):
    root = "/Users/wangqiang/Source/uci/"
    name_file = root+dataname+"/"+dataname+".names"
    data_file = root+dataname+"/"+dataname+".data"
    dataset = read_dataset(name_file, data_file, name=dataname)
    return dataset


def read_pics(file_dir,name):#set_minimum_fraction_of_weight_info_gain=0.1,grace_period=100,hoeffding_tie_threshold=0.06,split_confidence=0.2
    '''
    读取人脸图片数据
    :param file_dir:
    :param name:
    :return:
    '''
    pos_root = file_dir+"/pos"
    neg_root = file_dir+"/neg"
    pos_dirlist = os.listdir(pos_root)
    neg_dirlist = os.listdir(neg_root)
    im = Image.open(os.path.join(pos_root,pos_dirlist[0]))
    im_array = np.array(im)
    im_array = im_array.flatten()
    l = im_array.shape[0]
    headers = list(range(l))
    attributes = []
    #图片压缩为一维数组作为所有属性，所有的都是数字型
    for i in range(len(headers)):
        attributes.append(Attribute(str(headers[i]), att_type='Numeric'))

    #最后一维加一个label，是字符型
    headers.append('label')
    attributes.append(Attribute(str(headers[-1]), ["face","non-face"], 'Nominal'))
    class_index = len(headers) - 1
    instances = []
    for im_dir in pos_dirlist:
        im = Image.open(os.path.join(pos_root,im_dir))
        im_array = np.array(im)
        im_array = list(im_array.flatten())
        im_array.append("face")
        instances.append(im_array)
    for im_dir in neg_dirlist:
        im = Image.open(os.path.join(neg_root,im_dir))
        im_array = np.array(im)
        im_array = list(im_array.flatten())
        im_array.append("non-face")
        instances.append(im_array)
    random.shuffle(instances)
    dataset = Dataset(attributes, class_index, name=name)
    for inst in instances:
        inst[l] = int(attributes[l].index_of_value(str(inst[l])))#只有最后一个label是字符型的
        dataset.add(Instance(att_values=inst))
    return dataset

def dump_dataset(dataset,path):
    with open(path, 'wb') as f:
        pickle.dump(dataset, f)
def load_dataset( path):
        with open(path, 'rb')as f:
            dataset = pickle.load(f)
        return dataset

def read_minst_csv(filepath, name):#set_minimum_fraction_of_weight_info_gain=0.06,grace_period=100,hoeffding_tie_threshold=1.1,split_confidence=0.000001
    '''
    读取手写数字集数据
    :param filepath:
    :param name:
    :return:
    '''
    data = csv.reader(open(filepath, 'r'))
    l =0
    for d in data:
        l = len(d)
        break
    headers = list(range(l))
    headers[0] = "label"
    attributes = []
    # 第一位是label，是字符型
    labels = []
    for i in range(10):
        labels.append(str(i))
    attributes.append(Attribute(str(headers[0]), labels, 'Nominal'))
    # 图片压缩为一维数组作为所有属性，所有的都是数字型
    for i in range(1, len(headers)):
        attributes.append(Attribute(str(headers[i]), att_type='Numeric'))

    class_index = 0
    instances = []
    for d in data:
        instances.append(d)

    random.shuffle(instances)
    dataset = Dataset(attributes, class_index, name=name)
    for inst in instances:
        for i in range(len(inst)):
            inst[i] = int(inst[i])
        # inst[0] = int(attributes[0].index_of_value(str(inst[0])))  # 只有最后一个label是字符型的
        dataset.add(Instance(att_values=inst))
    return dataset


def main():
    # filename = 'dataset_file.csv'
    # dataset = open_dataset(filename, 1, probe_instances=10000)

    dataset = choic_dataset("letter")#("adult")
    vfdt = HoeffdingTree()
    print(dataset)
    # Set some of the algorithm parameters
    vfdt.set_grace_period(50)#计算分裂的周期
    vfdt.set_hoeffding_tie_threshold(1.2)
    vfdt.set_split_confidence(0.0001)
    # Split criterion, for now, can only be set on hoeffdingtree.py file.
    # This is only relevant when Information Gain is chosen as the split criterion
    vfdt.set_minimum_fraction_of_weight_info_gain(0.01)

    vfdt.build_classifier(dataset)
    
    # # Simulate a data stream
    # with open(filename) as f:
    #     stream = csv.reader(f)
    #     # Ignore the CSV headers
    #     next(stream)
    #     for item in stream:
    #         inst_values = list(item)
    #         for i in range(len(inst_values)):
    #             if dataset.attribute(index=i).type() == 'Nominal':
    #                 inst_values[i] = int(dataset.attribute(index=i)
    #                     .index_of_value(str(inst_values[i])))
    #             else:
    #                 inst_values[i] = float(inst_values[i])
    #         new_instance = Instance(att_values=inst_values)
    #         new_instance.set_dataset(dataset)
    #         vfdt.update_classifier(new_instance)
    print(vfdt)
    distubute = [0]*dataset.num_classes()
    count = 0
    for i in range(dataset.num_instances()):
        label = dataset.class_attribute().value(dataset.instance(i).class_value())
        distubute[dataset.instance(i).class_value()]+=1
        pre = vfdt.predict(dataset.instance(i))
        if label == pre:
            count+=1
    print("accucy:",count/dataset.num_instances())

def read_mix_train_pics(file_dir,number):
    '''

    :param file_dir:
    :param number:读取混合训练数据的个数
    :return:
    '''

    root = file_dir
    dirlist = os.listdir(root)
    im = Image.open(os.path.join(root,dirlist[0]))
    im_array = np.array(im)
    im_array = im_array.flatten()
    l = im_array.shape[0]
    headers = list(range(l))
    attributes = []
    #图片压缩为一维数组作为所有属性，所有的都是数字型
    for i in range(len(headers)):
        attributes.append(Attribute(str(headers[i]), att_type='Numeric'))

    #最后一维加一个label，是字符型
    headers.append('label')
    attributes.append(Attribute(str(headers[-1]), ["face","non-face"], 'Nominal'))
    class_index = len(headers) - 1
    instances = []
    for im_dir in dirlist[:number]:
        im = Image.open(os.path.join(root,im_dir))
        im_array = np.array(im)
        im_array = list(im_array.flatten())
        if 'n' in im_dir:
            im_array.append("non-face")
        else:
            im_array.append("face")
        instances.append(im_array)
    random.shuffle(instances)
    dataset = Dataset(attributes, class_index, name="train_data")
    for inst in instances:
        inst[l] = int(attributes[l].index_of_value(str(inst[l])))#只有最后一个label是字符型的
        dataset.add(Instance(att_values=inst))
    return dataset
def test(set_minimum_fraction_of_weight_info_gain=0.1,grace_period=100,hoeffding_tie_threshold=0.06,split_confidence=0.2):
    source = "/Users/wangqiang/Source"#"/home/wangqiang/Source"
    train_root =source+"/facedata/train"
    test_root = source+"/facedata/test"
    #train_dat a = read_pics(train_root, "train_set")
    train_data = read_mix_train_pics( source + "/facedata/mixed_train", 14000)
    test_data = read_pics(test_root, "test_set")
    # dump_dataset(train_data,"source/train.data")
    # dump_dataset(test_data, "source/test.data")
    # train_data = load_dataset("source/train.data")
    # test_data = load_dataset("source/test.data")
    # train_data = read_minst_csv("/Users/wangqiang/Source/minst/mnist_train.csv", "train_set")
    # test_data = read_minst_csv("/Users/wangqiang/Source/minst/mnist_test.csv", "test_set")

    print(train_data)
    print(train_data.get_distribute())
    # print(test_data.get_distribute())
    vfdt = HoeffdingTree()
    vfdt.set_grace_period(grace_period)#计算分裂的周期
    vfdt.set_hoeffding_tie_threshold(hoeffding_tie_threshold)
    vfdt.set_split_confidence(split_confidence)
    vfdt.set_minimum_fraction_of_weight_info_gain(set_minimum_fraction_of_weight_info_gain)
    vfdt.build_classifier(train_data)#,test_data
    # print(vfdt)
    vfdt.valuate_acc(test_data)

    pics = []
    for i in range(9):
        path = "source/" + str(i) + '.bmp'
        im = Image.open(path)
        im = im.resize((24, 24), Image.ANTIALIAS)
        im_array = np.array(im)
        im_array = list(im_array.flatten())
        ins = Instance(att_values=im_array)
        ins.set_dataset(test_data)
        pics.append(ins)
    for pic in pics:
        print(vfdt.predict(pic))
    vfdt.dump_mode()


if __name__ == '__main__':
    #main()
    test()




