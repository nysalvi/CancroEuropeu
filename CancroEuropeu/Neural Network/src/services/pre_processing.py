from torchvision import datasets, transforms, models
from ..model.neural_data import NeuralData
from ..utils.make_dataset import make_dataset
import os

def find_classes(self, directory: str) -> tuple[list[str], dict[str, int]]:
    labels = []        
    idx_label = {}
    for root, dirs, files in os.walk(directory):
        if dirs == []:            
            str_split = root.split('\\')[-1]
            pair = str_split.split(' - ') if len(str_split.split(' - ')) == 2 else False
            if pair:
                num, label = pair     
                if not label in labels:
                    labels.append(label)            
                idx_label.update({label : int(num)})
    return (labels, idx_label)

class PreProcessing:
    input_size: any
    dataPath: str

    def __init__(self, dataPath, height, width) -> None:
        self.input_size = (height,width)
        self.dataPath = dataPath
        pass

    def run(self) -> NeuralData:       
        data_transforms = {
        'train' : transforms.Compose([                
                transforms.RandomResizedCrop(self.input_size),
                #transforms.Resize(self.input_size),
                transforms.RandomHorizontalFlip(),
                transforms.RandomVerticalFlip(),
                #transforms.ColorJitter(brightness=0.05, contrast=0.05, hue=0.05),                                            
                transforms.ToTensor(),
                transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])                                                    
                ]),
        'dev' : transforms.Compose([                                    
            transforms.Resize(self.input_size),        
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])            
        ]),
        'test'   : transforms.Compose([
                transforms.Resize(self.input_size),        
                transforms.ToTensor(),
                transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
            ])
        }

        datasets.ImageFolder.find_classes = find_classes
        datasets.ImageFolder.make_dataset = make_dataset

        train_images=datasets.ImageFolder(self.dataPath + '/train', transform=data_transforms['train'])
        dev_images=datasets.ImageFolder(self.dataPath + '/dev', transform=data_transforms['dev'])
        test_images=datasets.ImageFolder(self.dataPath + '/test', transform=data_transforms['test'])

        return NeuralData(train_images, dev_images, test_images)
