from decimal import ROUND_CEILING
from albumentations.pytorch import ToTensorV2
from ..utils.make_dataset import make_dataset
from ..model.neural_data import NeuralData
from ..model.dataset import DataSet
from torchvision import transforms, datasets
from torch.nn import ModuleList, Sequential
import albumentations as A
import math
import os

def find_classes(self, directory: str) -> tuple[list[str], dict[str, int]]:
    labels = []        
    idx_label = {}
    for root, dirs, files in os.walk(directory):
        if dirs == []:            
            str_split = root.split(f'{os.sep}')[-1]
            pair = str_split.split(' - ') if len(str_split.split(' - ')) == 2 else False
            if pair:
                num, label = pair     
                if not label in labels:
                    labels.append(label)            
                idx_label.update({label : int(num)})
    return (labels, idx_label)
    
class PreProcessing:
    def __init__(self, dataPath, height, width) -> None:
        self.height = height
        self.width = width
        self.input_size = (height,width)
        self.dataPath = dataPath
        pass

    def run(self, train) -> NeuralData:       
        #1.1428571428571428571428571428571                    

        if train: 
            train = eval(train)            
        else:
            #train = A.Compose([
            #    A.Resize(self.height, self.width, always_apply=True), 
            #    A.ToFloat(max_value=255, always_apply=True),
            #    A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225], max_pixel_value=1.0, always_apply=True),
            #    ToTensorV2(transpose_mask=True, always_apply=True)
            #])
            train = transforms.Compose([
                transforms.Resize(self.input_size),        
                transforms.ToTensor(),
                transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])            
            ])        
        print(train)
        #dev = A.Compose([
        #    A.Resize(self.height, self.width, always_apply=True), 
        #    A.ToFloat(max_value=255, always_apply=True),
        #    A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225], max_pixel_value=1.0, always_apply=True),
        #    ToTensorV2(transpose_mask=True, always_apply=True)
        #])
        dev = transforms.Compose([                                    
            transforms.Resize(self.input_size),        
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])            
        ])  
        #test = A.Compose([
        #    A.Resize(self.height, self.width, always_apply=True), 
        #    A.ToFloat(max_value=255, always_apply=True),
        #    A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225], max_pixel_value=1.0, always_apply=True),
        #    ToTensorV2(transpose_mask=True, always_apply=True)
        #])
        test= transforms.Compose([                                    
            transforms.Resize(self.input_size),        
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])            
        ])

        data_transforms = {
            'train' : train,
            'dev' : dev,
            'test'   : test
        }
        datasets.ImageFolder.find_classes = find_classes
        datasets.ImageFolder.make_dataset = make_dataset

        train_images=datasets.ImageFolder(self.dataPath + f'{os.sep}train', transform=data_transforms['train'])
        dev_images=datasets.ImageFolder(self.dataPath + f'{os.sep}dev', transform=data_transforms['dev'])
        test_images=datasets.ImageFolder(self.dataPath + f'{os.sep}test', transform=data_transforms['test'])

        #train_images=DataSet(self.dataPath + f'{os.sep}train', transform=data_transforms['train'])
        #dev_images=DataSet(self.dataPath + f'{os.sep}dev', transform=data_transforms['dev'])
        #test_images= DataSet(self.dataPath + f'{os.sep}test', transform=data_transforms['test'])

        return NeuralData(train_images, dev_images, test_images)
