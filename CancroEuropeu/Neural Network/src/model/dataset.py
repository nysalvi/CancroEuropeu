import torch.utils.data as data_utils
from skimage import io
import numpy as np
import torch
import os

class DataSet(data_utils.Dataset):
    def create_dataset(self):
        sets = []        
        for root, dirs, files in os.walk(self.root_dir):
            if not dirs:              
                label = int(root.split(f'{os.sep}')[-1])
                labels = [label]*len(files)  
                self.classes.append(label)
                labels_data = np.array([files, labels])
                sets.append(labels_data)
        dataset = np.concatenate(sets, axis=1, dtype='object')        
        dataset = dataset.transpose(1, 0)
        dataset[:, 1] = dataset[:, 1].astype(np.int8)        
        return dataset    
        
    def __init__(self, root_dir, transform=None):        
        self.root_dir = root_dir        
        self.transform = transform
        self.classes = []
        self.__dfsize__ = 0
        for l in os.listdir(self.root_dir): 
            self.__dfsize__ += len(os.listdir(os.path.join(self.root_dir, l)))        
        self.dataset = self.create_dataset()        

    def __len__(self):
        return self.__dfsize__

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()
        name, label = self.dataset[idx]
        img_name = os.path.join(self.root_dir, str(label), name)
        image = io.imread(img_name)
        if self.transform:
            image = self.transform(image=image)
            image = image['image']        
        return image, label