import os
import numpy as np
import torchvision.transforms as transforms
from PIL import Image

root_dir = './../Datasets/python'

def find_classes(root_dir):
    img_lists = []
    for (root, dirs, files) in os.walk(root_dir):
        for file in files:
            if (file.endswith("png")):
                r = root.split('/')
                img_name = r[-2]
                char_num = r[-1]
                img_class = img_name+'/'+char_num
                img_lists.append((file, img_class, root))
    print("Found %d items: "% len(img_lists))
    return img_lists


def label_clasess(items):
    class_idx = {}
    count = 0
    for item in items:
        if item[1] not in class_idx:
            # assign this item a count = label
            class_idx[item[1]] = count
            count += 1
            
    print("Found %d classes: "%len(class_idx))
    return class_idx


img_items = find_classes(root_dir)
img_labels = label_clasess(img_items)

temp = {}
for img_name, img_class, img_path in img_items:
    img = '{}/{}'.format(img_path, img_name)
    label = img_labels[img_class]
    
    transform = transforms.Compose([lambda img: Image.open(img).convert('L'),
                              lambda img: img.resize((28,28)),
                              lambda img: np.reshape(img, (28,28,1)),
                              lambda img: np.transpose(img, [2,0,1]),
                              lambda img: img/255.
                              ])
    
    img = transform(img)
    if label in temp.keys():
        temp[label].append(img)
    else:
        temp[label] = [img]
        
print('Start generate omniglot.npy')
img_lists = []
for label, imgs in temp.items():
    img_lists.append(np.array(imgs))
img_lists = np.array(img_lists).astype(np.float32) # 20 images per class, 1623 classes in total

# Shape should be: (1623, 20, 1, 28, 28):
print('Data shape {}'.format(img_lists.shape))

np.save(os.path.join(root_dir, 'omniglot.npy'), img_lists)
    
        
    
    