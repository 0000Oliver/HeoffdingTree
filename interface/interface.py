import wx
# !/usr/bin/env python
"""Hello, wxPython! program."""
import sys
sys.path.append("..")
import wx
from main import *
import time
import threading

class Frame(wx.Frame):
    """Frame class that displays an image."""

    def __init__(self, image, parent=None, id=-1,
                 pos=wx.DefaultPosition,
                 title='Hello, wxPython!'):
        """Create a Frame instance and display image."""

        temp = image.ConvertToBitmap()
        size = temp.GetWidth(), temp.GetHeight()
        wx.Frame.__init__(self, parent, id, title, pos, size)
        self.bmp = wx.StaticBitmap(parent=self, bitmap=temp)


class App(wx.App):
    """Application class."""

    def OnInit(self):
        image = wx.Image('./source/1.jpg', wx.BITMAP_TYPE_JPEG)
        self.frame = Frame(image)
        self.frame.Show()
        self.SetTopWindow(self.frame)
        return True


def main():
    app = App()
    app.MainLoop()


if __name__ == '__main__':


    def start_train(event):
        thread_obj = threading.Thread(target=train)
        thread_obj.start()
    def train():


        vfdt = HoeffdingTree()
        vfdt.set_grace_period(grace_period)  # 计算分裂的周期
        vfdt.set_hoeffding_tie_threshold(hoeffding_tie_threshold)
        vfdt.set_split_confidence(split_confidence)
        vfdt.set_minimum_fraction_of_weight_info_gain(set_minimum_fraction_of_weight_info_gain)


        vfdt.reset()
        vfdt._header = train_data
        if vfdt._selected_split_metric is vfdt.GINI_SPLIT:
            vfdt._split_metric = GiniSplitMetric()
        else:
            vfdt._split_metric = InfoGainSplitMetric(vfdt._min_frac_weight_for_two_branches_gain)

        count = 0
        test_epoch = 100
        test_acc_list = []
        for i in range(train_data.num_instances()):
            vfdt.update_classifier(train_data.instance(i))
            count += 1

            if count % test_epoch == 0 and test_data != None:
                acc = vfdt.valuate_acc(test_data)
                test_acc_list.append(acc)
                pic_label =[]
                for pic in pics:
                    pic_label.append(vfdt.predict(pic))
                for i in range(9):
                    if pic_label[i]=='face':
                        image = wx.Image('../source/b'+str(i)+'.bmp').Rescale(200, 200).ConvertToBitmap()
                        img_boxs[i].SetBitmap(wx.BitmapFromImage(image))

                picnumber_text.SetLabel("图片数量:" + str(count))
                acc_text.SetLabel("准确率："+str(acc))
                # time.sleep(1)


        #vfdt.build_classifier(train_data)  # ,test_data
        # print(vfdt)
        vfdt.valuate_acc(test_data)



        vfdt.dump_mode()


    # def save(event):
    #     file = open(filename.GetValue(), 'w')
    #     file.write(contents.GetValue())
    #     file.close()


    app = wx.App()
    win = wx.Frame(None, title='Simple Editor', size=(650, 650))
    bkg = wx.Panel(win)

    loadButton = wx.Button(bkg, label='开始输入数据流')
    loadButton.Bind(wx.EVT_BUTTON, start_train)
    imgs = []
    img_boxs = []
    for i in range(9):
        img = wx.Image('../source/'+str(i)+'.bmp').Scale(200, 200).ConvertToBitmap()
        img_boxs.append(wx.StaticBitmap(bkg ,-1, img,(5,5)))




    #filename = wx.TextCtrl(bkg)
    #contents = wx.TextCtrl(bkg, style=wx.TE_MULTILINE | wx.HSCROLL)
    picnumber_text = wx.StaticText(bkg, label="图片数量：0",size = (210,25))
    acc_text = wx.StaticText(bkg, label="准确率：",size = (210,25))

    hbox = wx.BoxSizer()
    #hbox.Add(filename, proportion=1, flag=wx.EXPAND)
    hbox.Add(picnumber_text, proportion=1, flag=wx.LEFT)
    hbox.Add(acc_text, proportion=1, flag=wx.LEFT)
    hbox.Add(loadButton, proportion=0, flag=wx.LEFT, border=5)


    picbox = wx.GridSizer(cols=3, rows=3, vgap=5,hgap=5)
    for box in img_boxs:
        picbox.Add(box, 0, wx.LEFT, border=5)





    vbox = wx.BoxSizer(wx.VERTICAL)
    vbox.Add(hbox, proportion=0, flag=wx.EXPAND | wx.ALL, border=5)
    #vbox.Add(contents, proportion=1, flag=wx.EXPAND | wx.LEFT | wx.BOTTOM | wx.RIGHT, border=5)
    # vbox.Add(pichbox1,proportion =2,flag=wx.EXPAND | wx.ALL, border=5)
    # vbox.Add(pichbox2, proportion=3, flag=wx.EXPAND | wx.ALL, border=5)
    # vbox.Add(pichbox3, proportion=4, flag=wx.EXPAND | wx.ALL, border=5)
    vbox.Add(picbox,proportion =2,flag = wx.LEFT,border =5)


    bkg.SetSizer(vbox)
    win.Show()

    set_minimum_fraction_of_weight_info_gain = 0.1
    grace_period = 100
    hoeffding_tie_threshold = 0.06
    split_confidence = 0.2
    source = "/Users/wangqiang/Source"  # "/home/wangqiang/Source"
    train_root = source + "/facedata/train"
    test_root = source + "/facedata/test"
    train_data = read_pics(train_root, "train_set")
    test_data = read_pics(test_root, "test_set")


    pics = []
    for i in range(9):
        path = "../source/" + str(i) + '.bmp'
        im = Image.open(path)
        im = im.resize((24, 24), Image.ANTIALIAS)
        im_array = np.array(im)
        im_array = list(im_array.flatten())
        ins = Instance(att_values=im_array)
        ins.set_dataset(test_data)
        pics.append(ins)

    app.MainLoop()

