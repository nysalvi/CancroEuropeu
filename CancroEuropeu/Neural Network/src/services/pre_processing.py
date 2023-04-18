from torchvision import datasets, transforms
from src.model.neural_data import NeuralData

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
                transforms.Resize(self.input_size),
                transforms.RandomHorizontalFlip(),
                #transforms.ColorJitter(brightness=0.05, contrast=0.05, hue=0.05),                                            
                transforms.ToTensor(),
                #transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])                                                    
                ]),
        'test'   : transforms.Compose([
                transforms.Resize(self.input_size),        
                transforms.ToTensor(),
                #transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
            ])
        }

        train_images=datasets.ImageFolder(self.dataPath + '/train', transform=data_transforms['train'])
        test_images=datasets.ImageFolder(self.dataPath + '/test', transform=data_transforms['test'])

        return NeuralData(train_images, test_images)