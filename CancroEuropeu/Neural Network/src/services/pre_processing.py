from albumentations.pytorch import ToTensorV2
from ..utils.make_dataset import make_dataset
from ..model.neural_data import NeuralData
from ..model.dataset import DataSet
from torchvision import transforms
import albumentations as A
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

    def run(self, train_transform=None, dev_transform=None, test_transform=None) -> NeuralData:       
        train = A.Compose([            
            A.OneOf([
                A.Resize(self.height, self.width, p=0.2), 
                A.Compose([
                    A.Resize(256, 256), 
                    A.RandomResizedCrop(self.height, self.width, scale=(0.35, 0.95)),
                ], p=0.2),
                A.Compose([
                    A.Resize(256, 256), 
                    A.CenterCrop(self.height, self.width),
                ], p=0.2),
                A.Compose([
                    A.Resize(256, 256), 
                    A.RandomCrop(self.height, self.width),
                ], p=0.2),
                A.Compose([
                    A.Resize(self.height, self.width), 
                    A.CropAndPad(px=(15, 10, 15, 10), keep_size=True, sample_independently=True)
                ], p=0.2)
            ], p=1),
            A.OneOf([
                A.Defocus(radius=(2, 4), alias_blur=(0.1, 0.3), p=0.33), 
                A.MedianBlur(blur_limit=3, p=0.33),
                A.MotionBlur(blur_limit=11, allow_shifted=True, p=0.33),
            ], p=0.75),
            A.ColorJitter(brightness=0.1, contrast=0.1, saturation=0.1, hue=0.1, p=0.5),
            A.RandomBrightnessContrast(brightness_limit=(-0.25, 0.15), contrast_limit=(-0.25, 0.3), brightness_by_max=False, p=0.5),
            A.HorizontalFlip(p=0.5),
            A.VerticalFlip(p=0.5),
            A.ToFloat(max_value=255, always_apply=True),
            A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225], max_pixel_value=1.0, always_apply=True),
            ToTensorV2(transpose_mask=True, always_apply=True),
        ])

        dev = A.Compose([            
            A.OneOf([
                A.Resize(self.height, self.width, p=0.2), 
                A.Compose([
                    A.Resize(256, 256), 
                    A.RandomResizedCrop(self.height, self.width, scale=(0.35, 0.95)),
                ], p=0.2),
                A.Compose([
                    A.Resize(256, 256), 
                    A.CenterCrop(self.height, self.width),
                ], p=0.2),
                A.Compose([
                    A.Resize(256, 256), 
                    A.RandomCrop(self.height, self.width),
                ], p=0.2),
                A.Compose([
                    A.Resize(self.height, self.width), 
                    A.CropAndPad(px=(15, 10, 15, 10), keep_size=True, sample_independently=True)
                ], p=0.2)
            ], p=1),
            A.OneOf([
                A.Defocus(radius=(2, 4), alias_blur=(0.1, 0.3), p=0.33), 
                A.MedianBlur(blur_limit=3, p=0.33),
                A.MotionBlur(blur_limit=11, allow_shifted=True, p=0.33),
            ], p=0.75),
            A.ColorJitter(brightness=0.1, contrast=0.1, saturation=0.1, hue=0.1, p=0.5),
            A.RandomBrightnessContrast(brightness_limit=(-0.25, 0.15), contrast_limit=(-0.25, 0.3), brightness_by_max=False, p=0.5),
            A.HorizontalFlip(p=0.5),
            A.VerticalFlip(p=0.5),
            A.ToFloat(max_value=255, always_apply=True),
            A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225], max_pixel_value=1.0, always_apply=True),
            ToTensorV2(transpose_mask=True, always_apply=True),
        ])

        #if train_transform:
        #    train = eval(train_transform)
        #else:
        #    train = transforms.Compose([                
        #        transforms.Resize(self.input_size),
        #        transforms.ToTensor(),
        #        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        #    ])
        #
        #if dev_transform:
        #    dev = eval(dev_transform)
        #else:
        #    dev = transforms.Compose([                                    
        #        transforms.Resize(self.input_size),        
        #        transforms.ToTensor(),
        #        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])            
        #    ])
        #

        if test_transform:
            test = eval(test_transform)
        else:
            #test = transforms.Compose([                                    
            #    transforms.Resize(self.input_size),        
            #    transforms.ToTensor(),
            #    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])            
            #])
            test = A.Compose([
                A.Resize(self.height, self.width, always_apply=True), 
                A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225], max_pixel_value=255.0, always_apply=True),
                A.ToFloat(max_value=255, always_apply=True),
                ToTensorV2(transpose_mask=True, always_apply=True)
            ])

        data_transforms = {
            'train' : train,
            'dev' : dev,
            'test'   : test
        }
        #datasets.ImageFolder.find_classes = find_classes
        #datasets.ImageFolder.make_dataset = make_dataset

        #train_images=datasets.ImageFolder(self.dataPath + f'{os.sep}train', transform=data_transforms['train'])
        #dev_images=datasets.ImageFolder(self.dataPath + f'{os.sep}dev', transform=data_transforms['dev'])
        #test_images=datasets.ImageFolder(self.dataPath + f'{os.sep}test', transform=data_transforms['test'])

        train_images=DataSet(self.dataPath + f'{os.sep}train', transform=data_transforms['train'])
        dev_images=DataSet(self.dataPath + f'{os.sep}dev', transform=data_transforms['dev'])
        test_images= DataSet(self.dataPath + f'{os.sep}test', transform=data_transforms['test'])

        return NeuralData(train_images, dev_images, test_images)
