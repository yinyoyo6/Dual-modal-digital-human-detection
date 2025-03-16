import warnings
warnings.filterwarnings("ignore")
import torch
from PIL import Image
from torchvision import datasets, models, transforms,utils
import torch.nn as nn
import numpy as np
import random
import os
import torchvision
from tqdm import tqdm
import matplotlib.pyplot as plt
from sklearn.metrics import classification_report,precision_score, recall_score,f1_score
from models.vgg import VGG
from models.densenet import DenseNet121
from models.googlenet import GoogLeNet
from models.lenet import LeNet
from models.mobilenet import MobileNet
from models.resnet import ResNet50


#设置随机种子
def setup_seed(seed):
   torch.manual_seed(seed)
   torch.cuda.manual_seed(seed)
   torch.cuda.manual_seed_all(seed)
   np.random.seed(seed)
   random.seed(seed)
   torch.backends.cudnn.benchmark = False
   torch.backends.cudnn.deterministic = True
   torch.backends.cudnn.enabled = True
   os.environ['PYTHONHASHSEED'] = str(seed)
setup_seed(20)

root = './logs/'

# Hyper parameters
###可选择的model模型有:VGG\DenseNet121\LeNet\GoogLeNet\MobileNet\ResNet50
model = 'ResNet50'
num_epochs = 5    #循环次数
batch_size = 16 #每次投喂数据量
learning_rate = 0.00005   #学习率
momentum = 0.9  #变化率


def draw_test_process1(title, iters, label_cost):
    plt.figure()
    plt.title(title, fontsize=24)
    plt.plot(iters, label_cost, '-', label=label_cost)
    plt.ylabel(str(title))
    plt.ylim([0, 1.05])
    plt.xlabel('epoch')
    # plt.show()
    plt.savefig('./result/'+title+'.jpg')



def evaluate(model, data_loader, device, epoch):

    precison1 = []  #计算平均精度
    recall1 = []    #计算平均召回率
    f1score = []    #计算平均得分


    loss_function = torch.nn.CrossEntropyLoss()

    model.eval()

    accu_num = torch.zeros(1).to(device)   # 累计预测正确的样本数
    accu_loss = torch.zeros(1).to(device)  # 累计损失

    sample_num = 0
    data_loader = tqdm(data_loader, file=sys.stdout)
    for step, data in enumerate(data_loader):
        images, labels = data
        sample_num += images.shape[0]

        pred = model(images.to(device))
        pred_classes = torch.max(pred, dim=1)[1]
        accu_num += torch.eq(pred_classes, labels.to(device)).sum()

        loss = loss_function(pred, labels.to(device))
        accu_loss += loss

        data_loader.desc = "[valid epoch {}] loss: {:.3f}, acc: {:.3f}".format(epoch,
                                                                               accu_loss.item() / (step + 1),
                                                                               accu_num.item() / sample_num)
        precison1.append(precision_score(y_true=labels.tolist(), y_pred=pred_classes.tolist()))
        recall1.append(recall_score(y_true=labels.tolist(), y_pred=pred_classes.tolist()))
        f1score.append(f1_score(y_true=labels.tolist(), y_pred=pred_classes.tolist()))

    print('\n')
    w = 0
    for j in range(len(precison1)):
        w += precison1[j]
    print("precision:",w/len(precison1))
    w = 0
    for j in range(len(recall1)):
        w += recall1[j]
    print("recall:",w/len(recall1))
    w = 0
    for j in range(len(f1score)):
        w += f1score[j]
    print("f1score:",w/len(f1score))

    return accu_loss.item() / (step + 1), accu_num.item() / sample_num


class MyDataset(torch.utils.data.Dataset):  # 创建自己的类：MyDataset,这个类是继承的torch.utils.data.Dataset
    def __init__(self, datatxt, transform=None, target_transform=None):  # 初始化一些需要传入的参数
        super(MyDataset, self).__init__()
        fh = open(datatxt, 'r')  # 按照传入的路径和txt文本参数，打开这个文本，并读取内容
        imgs = []  # 创建一个名为img的空列表，一会儿用来装东西
        for line in fh:  # 按行循环txt文本中的内容
            line = line.rstrip()  # 删除 本行string 字符串末尾的指定字符，这个方法的详细介绍自己查询python
            words = line.split()  # 通过指定分隔符对字符串进行切片，默认为所有的空字符，包括空格、换行、制表符等
            imgs.append((line[:-2], int(words[-1])))  # 把txt里的内容读入imgs列表保存，具体是words几要看txt内容而定
            # 很显然，words[0]是图片信息，words[1]是lable
        self.imgs = imgs
        self.transform = transform
        self.target_transform = target_transform

    def __getitem__(self, index):  # 这个方法是必须要有的，用于按照索引读取每个元素的具体内容
        fn, label = self.imgs[index]  # fn是图片path #fn和label分别获得imgs[index]也即是刚才每行中word[0]和word[1]的信息
        img = Image.open(fn).convert('RGB')  # 按照path读入图片from PIL import Image # 按照路径读取图片 彩色图片则为RGB
        img = img.resize((32,32))

        if self.transform is not None:
            img = self.transform(img)  # 是否进行transform
        return img, label  # return很关键，return回哪些内容，那么我们在训练时循环读取每个batch时，就能获得哪些内容

    def __len__(self):  # 这个函数也必须要写，它返回的是数据集的长度，也就是多少张图片，要和loader的长度作区分
        return len(self.imgs)

# 根据自己定义的那个类MyDataset来创建数据集！注意是数据集！而不是loader迭代器
train_data = MyDataset(datatxt=root + 'train.txt', transform=transforms.ToTensor())
test_data = MyDataset(datatxt=root + 'test.txt', transform=transforms.ToTensor())

#然后就是调用DataLoader和刚刚创建的数据集，来创建dataloader，这里提一句，loader的长度是有多少个batch，所以和batch_size有关
train_loader = torch.utils.data.DataLoader(dataset=train_data, batch_size = batch_size, shuffle=True)
test_loader = torch.utils.data.DataLoader(dataset=test_data, batch_size = batch_size, shuffle=False)

print("PyTorch Version: ",torch.__version__)
print("Torchvision Version: ",torchvision.__version__)

#开启gpu加速
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
# VGG\DenseNet121\LeNet\GoogLeNet\MobileNet\ResNet50
if model == 'VGG':
    net = net = VGG('VGG11')
elif model == 'DenseNet121':
    net = DenseNet121()
elif model == 'LeNet':
    net = LeNet()
elif model == 'GoogLeNet':
    net = GoogLeNet()
elif model == 'MobileNet':
    net = MobileNet()
elif model == 'ResNet50':
    net = ResNet50()


net = net.to(device)

# Loss and optimizer
criterion = nn.CrossEntropyLoss()
# optimizer = optim.SGD(net.parameters(), lr=learning_rate, momentum = momentum )
optimizer=torch.optim.Adam(net.parameters(),lr=learning_rate,betas=(0.9,0.999))

loss22 = []
acc22 = []
precison1 = []  #计算平均精度
recall1 = []    #计算平均召回率
f1score = []    #计算平均得分

# Train the model
total_step = len(train_loader)
for epoch in range(num_epochs):
    running_loss = 0.0
    train_loader = tqdm(train_loader)
    for i, (images, labels) in enumerate(train_loader):
        images = images.to(device)
        labels = labels.to(device)
        optimizer.zero_grad()
        # Forward pass
        outputs = net(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

    net.eval()
    test_loss = 0.
    test_acc = 0.
    y_true = []
    y_pred = []
    with torch.no_grad():   #这里告诉pytorch运算时不需计算图的
        # test
        total_correct = 0
        total_num = 0
        for x, label in test_loader:   #获取测试集数据
            # [b, 3, 32, 32]
            # [b]
            y_true += label
            x, label = x.to(device), label.to(device)   #调用cuda
            # [b, 10]
            logits = net(x)
            # [b]
            pred = logits.argmax(dim=1)  #在第2个维度上索引最大的值的下标
            y_pred += pred
            # [b] vs [b] => scalar tensor  比较预测值与真实值预测对的数量 eq是否相等
            correct = torch.eq(pred, label).float().sum().item()
            total_correct += correct
            total_num += x.size(0)  #统计输入总数
            # print(correct)
    # acc = total_correct / total_num  #计算平均准确率
    y_true = torch.stack(y_true)
    y_pred = torch.stack(y_pred)
    import os
    list_names = os.listdir('./data')
    print(classification_report(y_true.cpu().numpy(),y_pred.cpu().numpy(),digits=5,labels=range(len(list_names)),target_names=list_names))

    precison1.append(precision_score(y_true.cpu().numpy(),y_pred.cpu().numpy(),pos_label='positive',average='weighted'))
    recall1.append(recall_score(y_true.cpu().numpy(),y_pred.cpu().numpy(),pos_label='positive',average='weighted'))
    f1score.append(f1_score(y_true.cpu().numpy(),y_pred.cpu().numpy(),pos_label='positive',average='weighted'))




    torch.save(net, './logs/model.ckpt')



iters = np.arange(1,num_epochs+1,1)
draw_test_process1(str(model)+'_precison', iters, precison1)
draw_test_process1(str(model)+'_recall', iters, recall1)
draw_test_process1(str(model)+'_f1score', iters, f1score)