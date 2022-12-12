import torch
import numpy as np
import os

root_dir = './../Datasets/python'

# n_way = 5
# k_spt = 1  ## support data 的个数
# k_query = 15 ## query data 的个数
# imgsz = 28
# resize = imgsz
# task_num = 8
# batch_size = task_num

def load_data_cache(dataset):
    """
    Collects several batches data for N-shot learning
    :param dataset: [cls_num, 20, 84, 84, 1]
    :return: A list with [support_set_x, support_set_y, target_x, target_y] ready to be fed to our networks
    """
    n_way = 5
    k_spt = 1  ## support data 的个数
    k_query = 15 ## query data 的个数
    imgsz = 28
    resize = imgsz
    task_num = 8
    batch_size = task_num
    
    # 5 way 1 shot: 5 * 1
    setsz = k_spt*n_way
    querysz = k_query*n_way
    data_cache = []

    # print('preload next 10 caches of batch_size of batch.')
    for sample in range(10):
        
        x_spts, x_qrys, y_spts, y_qrys = [], [], [], []
        for i in range(batch_size):
            x_spt, x_qry, y_spt, y_qry = [], [], [], []
            selected_cls = np.random.choice(dataset.shape[0], n_way, replace = False)
            
            for j, cur_class in enumerate(selected_cls):
                selected_img = np.random.choice(20, k_spt + k_query, replace= False)
                
                x_spt.append(dataset[cur_class][selected_img[:k_spt]])
                x_qry.append(dataset[cur_class][selected_img[k_spt:]])
                y_spt.append([j for _ in range(k_spt)])
                y_qry.append([j for _ in range(k_query)])
        
        
            # shuffle inside a batch
            perm = np.random.permutation(n_way*k_spt)   
            x_spt = np.array(x_spt).reshape(n_way*k_spt, 1, resize, resize)[perm]
            y_spt = np.array(y_spt).reshape(n_way*k_spt)[perm]
            perm = np.random.permutation(n_way*k_query)   
            x_qry = np.array(x_qry).reshape(n_way*k_query, 1, resize, resize)[perm]
            y_qry = np.array(y_qry).reshape(n_way*k_query)[perm]
            
            # append [sptsz, 1, 84, 84] => [batch_size, setsz, 1, 84, 84]
            x_spts.append(x_spt)
            y_spts.append(y_spt)
            x_qrys.append(x_qry)
            y_qrys.append(y_qry)
            
        # [batch_size, setsz = n_way*k_spt, 1, resize, resize]
        x_spts = np.array(x_spts).astype(np.float32).reshape(batch_size, setsz, 1, resize, resize)
        y_spts = np.array(y_spts).astype(np.int).reshape(batch_size, setsz)
        # [batch_size, qrysz = n_way*k_query, 1, resize, resize]
        x_qrys = np.array(x_qrys).astype(np.float32).reshape(batch_size, querysz, 1, resize, resize)
        y_qrys = np.array(y_qrys).astype(np.int).reshape(batch_size, querysz)
        
        data_cache.append([x_spts, y_spts, x_qrys, y_qrys])
        
    return data_cache

# img_lists = np.load(os.path.join(root_dir, 'omniglot.npy'))
# x_train = img_lists[:1200]
# x_test = img_lists[1200:]
# num_classes = img_lists.shape[0]
# datasets = {'train': x_train, 'test': x_test}

# indexes = {"train": 0, "test": 0}
# # datasets = {"train": x_train, "test": x_test}
# print("DB: train", x_train.shape, "test", x_test.shape)


# # current eopoch data cached
# datasets_cache = {"train": load_data_cache(x_train), "test": load_data_cache(x_test)}


# def next(mode='train'):
#     """
#     Gets next batch from the dataset with name.
#     :param mode: The name of the splitting (one of "train", "val", "test")
#     :return:
#     """
    
#     if indexes[mode] >= len(datasets_cache[mode]):
#         indexes[mode] = 0
#         datasets_cache[mode] = load_data_cache(datasets[mode])
    
#     next_batch = datasets_cache[mode][indexes[mode]]
#     indexes[mode] += 1
    
#     return next_batch

            