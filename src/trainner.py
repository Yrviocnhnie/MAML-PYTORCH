import torch
import numpy as np
import time
import os

from load_cache import load_data_cache
from models.MetaLearner import MetaLearner

root_dir = './../Datasets/python'

device = torch.device('cuda')
meta = MetaLearner().to(device)


# img_lists = np.load(os.path.join(root_dir, 'omniglot.npy'))
img_lists = np.load('/home/richie/MAML-Pytorch/Datasets/python/omniglot.npy')
x_train = img_lists[:1200]
x_test = img_lists[1200:]
num_classes = img_lists.shape[0]
datasets = {'train': x_train, 'test': x_test}
indexes = {"train": 0, "test": 0}
# datasets = {"train": x_train, "test": x_test}
print("DB: train", x_train.shape, "test", x_test.shape)

# current eopoch data cached
datasets_cache = {"train": load_data_cache(x_train), "test": load_data_cache(x_test)}

def next(mode='train'):
    """
    Gets next batch from the dataset with name.
    :param mode: The name of the splitting (one of "train", "val", "test")
    :return:
    """
    
    if indexes[mode] >= len(datasets_cache[mode]):
        indexes[mode] = 0
        datasets_cache[mode] = load_data_cache(datasets[mode])
    
    next_batch = datasets_cache[mode][indexes[mode]]
    indexes[mode] += 1
    
    return next_batch

task_num = 8
epochs = 60000
for step in range(epochs):
    start = time.time()
    x_spt, y_spt, x_qry, y_qry = next('train')
    x_spt = torch.from_numpy(x_spt).to(device, dtype=torch.float32)
    y_spt = torch.from_numpy(y_spt).to(device)
    x_qry = torch.from_numpy(x_qry).to(device, dtype=torch.float32)
    y_qry = torch.from_numpy(y_qry).to(device)
    accs, loss = meta(x_spt, y_spt, x_qry, y_qry)
    end = time.time()
    
    if step % 100 == 0:
        print("epoch:" ,step)
        print(accs)
        print(loss)
        
    if step % 1000 == 0:
        accs = []
        for _ in range(1000//task_num):
            x_spt, y_spt, x_qry, y_qry = next('test')
            x_spt = torch.from_numpy(x_spt).to(device, dtype=torch.float32)
            y_spt = torch.from_numpy(y_spt).to(device)
            x_qry = torch.from_numpy(x_qry).to(device, dtype=torch.float32)
            y_qry = torch.from_numpy(y_qry).to(device)
            
            for x_spt_1, y_spt_1, x_qry_1, y_qry_1 in zip(x_spt, y_spt, x_qry, y_qry):
                test_acc = meta.finetuning(x_spt_1, y_spt_1, x_qry_1, y_qry_1)
                accs.append(test_acc)
                
        print('在mean process之前：',np.array(accs).shape)
        accs = np.array(accs).mean(axis=0).astype(np.float16)
        print('测试集准确率:',accs)