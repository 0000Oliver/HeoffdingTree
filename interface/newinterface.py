import wx
import os
import sys
sys.path.append("..")
from hoeffdingtree import *
from PIL import Image
import numpy as np
import random


def read_test_pics(file_dir,label):
    '''

    :param file_dir:
    :param label: 为1代表正样本  为0代表副样本
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
    for im_dir in dirlist:
        im = Image.open(os.path.join(root,im_dir))
        im_array = np.array(im)
        im_array = list(im_array.flatten())
        if label==1:
            im_array.append("face")
        else:
            im_array.append("non-face")
        instances.append(im_array)
    random.shuffle(instances)
    dataset = Dataset(attributes, class_index, name=str(label))
    for inst in instances:
        inst[l] = int(attributes[l].index_of_value(str(inst[l])))#只有最后一个label是字符型的
        dataset.add(Instance(att_values=inst))
    return dataset

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


class FileDrop(wx.FileDropTarget):
    def __init__(self, gridsizer,traindata_number):
        wx.FileDropTarget.__init__(self)
        self.gridsizer = gridsizer
        self.traindata_number=traindata_number

    def OnDropFiles(self, x, y, filePath):  # 当文件被拖入grid后，会调用此方法
        pics = []
        for file in filePath:
            im = Image.open(file)
            im_array = np.array(im)
            im_array = list(im_array.flatten())
            if 'n.bmp' in file:
                im_array.append(1)
            else:
                im_array.append(0)
            ins = Instance(att_values=im_array)
            ins.set_dataset(train_mixed_data)
            pics.append(ins)

            img = wx.Image(file).ConvertToBitmap()
            imgbox = wx.StaticBitmap(bkg, -1, img)
            self.gridsizer.Add(imgbox, 0, wx.LEFT, border=1)

        train(pics)

        for i in range(test_pos_picbox.GetItemCount()):
            test_pos_picbox.Remove(0)
        for i in range(test_neg_picbox.GetItemCount()):
            test_neg_picbox.Remove(0)
        self.traindata_number +=len(filePath)
        train_data_text.SetLabel("训练集：" + str(self.traindata_number))
        for b_i in img_boxs_instances:
            pre = vfdt.predict(b_i[1])
            if pre =="face":
                test_pos_picbox.Add(b_i[0], 0, wx.LEFT, border=1)
            else:
                test_neg_picbox.Add(b_i[0], 0, wx.LEFT, border=1)
        test_pos_picbox.Layout()
        test_neg_picbox.Layout()
        # hbox3.Remove(0)
        # hbox3.Remove(0)
        hbox3.Layout()
        self.gridsizer.Layout()
        vbox.Layout()
        bkg.SetSizer(vbox)
        bkg.SetScrollbars(1, 1, 600, vbox.Size[1]+500)
        return True




def train(pics):
        for pic in pics:
            vfdt.update_classifier(pic)

        pos_acc = vfdt.valuate_acc(test_pos_data)
        neg_acc = vfdt.valuate_acc(test_neg_data)
        testdata_pos_acc.SetLabel("训练准确率："+str(pos_acc))
        testdata_neg_acc.SetLabel("训练准确率："+str(neg_acc))



source = "/Users/wangqiang/Source"  # "/home/wangqiang/Source"
train_root = source + "/facedata/mixed_train"
test_root = source + "/facedata/test"
pos_dirlist = os.listdir(test_root+"/pos")
neg_dirlist = os.listdir(test_root+"/neg")
train_dirlist = os.listdir(train_root)

test_pos_data = read_test_pics(test_root+"/pos",1)
test_neg_data = read_test_pics(test_root+"/neg",0)

train_mixed_data = read_mix_train_pics(train_root,130)


#初始化 并加入几张图片训练一个模型
set_minimum_fraction_of_weight_info_gain = 0.1
grace_period = 5
hoeffding_tie_threshold = 0.06
split_confidence = 0.2
vfdt = HoeffdingTree()
vfdt.set_grace_period(grace_period)  # 计算分裂的周期
vfdt.set_hoeffding_tie_threshold(hoeffding_tie_threshold)
vfdt.set_split_confidence(split_confidence)
vfdt.set_minimum_fraction_of_weight_info_gain(set_minimum_fraction_of_weight_info_gain)
vfdt.reset()
vfdt._header = train_mixed_data
if vfdt._selected_split_metric is vfdt.GINI_SPLIT:
    vfdt._split_metric = GiniSplitMetric()
else:
    vfdt._split_metric = InfoGainSplitMetric(vfdt._min_frac_weight_for_two_branches_gain)

count = 0
test_epoch = 100
test_acc_list = []
vfdt.build_classifier(train_mixed_data)


app = wx.App()
win = wx.Frame(None, title='Simple Editor', size=(600, 650))


bkg = wx.ScrolledWindow(win, id=-1, pos=wx.DefaultPosition,
        size=wx.DefaultSize, style=  wx.HSCROLL |wx.VSCROLL,
        name="scrolledWindow")#wx.DefaultSize
bkg.SetScrollbars(1, 1, 600, 1000)
# bkg = wx.Panel(win)


#loadButton = wx.Button(bkg, label='开始输入数据流')
#loadButton.Bind(wx.EVT_BUTTON, start_train)
testimg_pos_boxs = []
img_boxs_instances = []
for i in range(45):
    img_dir = os.path.join(test_root+"/pos",pos_dirlist[i])

    im = Image.open(img_dir)
    im_array = np.array(im)
    im_array = list(im_array.flatten())
    im_array.append(0)
    ins = Instance(att_values=im_array)
    ins.set_dataset(train_mixed_data)

    img = wx.Image(img_dir).ConvertToBitmap()
    #testimg_pos_boxs.append(wx.StaticBitmap(bkg, -1, img))
    img_boxs_instances.append((wx.StaticBitmap(bkg, -1, img),ins))
testimg_neg_boxs = []
for i in range(45):
    img_dir = os.path.join(test_root+"/neg",neg_dirlist[i])

    im = Image.open(img_dir)
    im_array = np.array(im)
    im_array = list(im_array.flatten())
    im_array.append(0)
    ins = Instance(att_values=im_array)
    ins.set_dataset(train_mixed_data)

    img = wx.Image(img_dir).ConvertToBitmap()
    #testimg_neg_boxs.append(wx.StaticBitmap(bkg, -1, img))

    img_boxs_instances.append((wx.StaticBitmap(bkg, -1, img), ins))

random.shuffle(img_boxs_instances)
# filename = wx.TextCtrl(bkg)
# contents = wx.TextCtrl(bkg, style=wx.TE_MULTILINE | wx.HSCROLL)

pos_acc = vfdt.valuate_acc(test_pos_data)
neg_acc = vfdt.valuate_acc(test_neg_data)
testdata_text = wx.StaticText(bkg, label="测试集")#size=(210, 25)
testdata_pos_text = wx.StaticText(bkg, label="正样本" ,style=wx.ALIGN_CENTER,size=(300, 25))
testdata_neg_text = wx.StaticText(bkg, label="负样本", style=wx.ALIGN_CENTER,size=(300, 25))
testdata_pos_acc = wx.StaticText(bkg, label="测试准确率：0.5",style=wx.ALIGN_CENTER,size=(300, 25))
testdata_neg_acc = wx.StaticText(bkg, label="测试准确率：0.5",style=wx.ALIGN_CENTER,size=(300, 25))

test_pos_picbox = wx.FlexGridSizer(cols=9, vgap=2, hgap=2)
for b_i in img_boxs_instances[:45]:
    test_pos_picbox.Add(b_i[0], 0, wx.LEFT, border=1)
# for box in testimg_pos_boxs:
#     test_pos_picbox.Add(box, 0, wx.LEFT, border=1)
test_neg_picbox = wx.FlexGridSizer(cols=9,vgap=2, hgap=2)
for b_i in img_boxs_instances[45:]:
    test_neg_picbox.Add(b_i[0], 0, wx.LEFT, border=1)
# for box in testimg_neg_boxs:
#     test_neg_picbox.Add(box, 0, wx.LEFT, border=1)

train_data_text = wx.StaticText(bkg, label="训练集:10",style=wx.ALIGN_CENTER,size=(300, 25))
traindata_number =10
#taindata_number = wx.StaticText(bkg, label="训练样本数：0")


train_pic_box = wx.FlexGridSizer(cols=20, vgap=2, hgap=2)
fileDrop = FileDrop(train_pic_box,traindata_number)  # 第1步，创建FileDrop对象，并把grid传给初始化函数
win.SetDropTarget(fileDrop)  # 第2步，调用grid的SetDropTarget函数，并把FileDrop对象传给它
train_boxs = []
for i in range(10):
    img = wx.Image(os.path.join(train_root,train_dirlist[i])).ConvertToBitmap()
    train_boxs.append(wx.StaticBitmap(bkg, -1, img))
for box in train_boxs:
    train_pic_box.Add(box, 0, wx.LEFT, border=1)
train_pic_box.AddGrowableRow(0)
train_pic_box.AddGrowableCol(0)



hbox1 = wx.BoxSizer()
hbox2 = wx.BoxSizer()
hbox3 = wx.BoxSizer()
hbox4 =wx.BoxSizer()


# hbox.Add(filename, proportion=1, flag=wx.EXPAND)
hbox1.Add(testdata_text, proportion=1, flag=wx.ALIGN_CENTER)
hbox2.Add(testdata_pos_text, proportion=1, flag=wx.ALIGN_CENTER, border=10)
hbox2.Add(testdata_neg_text, proportion=0, flag=wx.ALIGN_CENTER, border=10)


hbox3.Add(test_pos_picbox, proportion=0,flag=wx.ALL|wx.ALIGN_CENTER,border =10)
hbox3.Add(test_neg_picbox, proportion=1,flag=wx.ALL|wx.ALIGN_CENTER,border =10)
hbox4.Add(testdata_pos_acc, proportion=1, flag=wx.ALIGN_CENTER)
hbox4.Add(testdata_neg_acc, proportion=0, flag=wx.ALIGN_CENTER)

vbox = wx.BoxSizer(wx.VERTICAL)
vbox.Add(hbox1, proportion=0, flag=wx.ALIGN_CENTER , border=5)
vbox.Add(hbox2, proportion=0, flag=wx.ALIGN_CENTER | wx.ALL, border=5)
# vbox.Add(contents, proportion=1, flag=wx.EXPAND | wx.LEFT | wx.BOTTOM | wx.RIGHT, border=5)
# vbox.Add(pichbox1,proportion =2,flag=wx.EXPAND | wx.ALL, border=5)
# vbox.Add(pichbox2, proportion=3, flag=wx.EXPAND | wx.ALL, border=5)
# vbox.Add(pichbox3, proportion=4, flag=wx.EXPAND | wx.ALL, border=5)
vbox.Add(hbox3, proportion=0, flag=wx.ALL|wx.ALIGN_CENTER, border=5)
vbox.Add(hbox4, proportion=0, flag=wx.ALL|wx.ALIGN_CENTER, border=5)
vbox.Add(train_data_text,proportion=0, flag=wx.ALL|wx.ALIGN_CENTER, border=5)
vbox.Add(train_pic_box,proportion=0, flag=wx.ALL|wx.ALIGN_CENTER, border=5)
#vbox.Add(scroll_panel,proportion=0, flag=wx.ALL|wx.ALIGN_CENTER, border=5)
# vbox.Add(taindata_number,proportion=0, flag=wx.ALL|wx.ALIGN_CENTER, border=5)

bkg.SetSizer(vbox)
win.Show()


app.MainLoop()





