from torchvision import datasets, transforms, models
from ..model.neural_data import NeuralData
from ..utils.make_dataset import make_dataset
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
        self.input_size = (height,width)
        self.dataPath = dataPath
        pass

    def run(self, train_transform=None, dev_transform=None, test_transform=None) -> NeuralData:       

        if train_transform:
            train = eval(train_transform)
        else:
            train = transforms.Compose([                
                transforms.Resize(self.input_size),
                transforms.ToTensor(),
                transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
            ])

        if dev_transform:
            dev = eval(dev_transform)
        else:
            dev = transforms.Compose([                                    
                transforms.Resize(self.input_size),        
                transforms.ToTensor(),
                transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])            
            ])

        if test_transform:
            test = eval(test_transform)
        else:
            test = transforms.Compose([                                    
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

        return NeuralData(train_images, dev_images, test_images)
