import os
import random

def setup_seed(seed):
    random.seed(seed)
setup_seed(20)

b = 0
dir = './data/'

files = os.listdir(dir)         
files.sort()

train = open('./logs/train.txt', 'w')
test = open('./logs/test.txt', 'w')
a = 0
a1 = 0
files = os.listdir('./data')
while(b < len(files)):
    label = a 
    ss = './data/' + str(files[b]) + '/'
    pics = os.listdir(ss) 
    i = 1
    train_percent = 0.8  

    num = len(pics) 
    list = range(num) 
    train_num = int(num * train_percent)  
    train_sample = random.sample(list, train_num)  
    test_num = num - train_num    

    for i in list: 
        name = str(dir) + str(files[b]) + '/' + pics[i] + ' ' + str(int(label)) + '\n'  
        if i in train_sample:  
            train.write(name) 
        else:
            test.write(name)  
    a = a + 1
    b = b + 1
train.close()  
test.close()
