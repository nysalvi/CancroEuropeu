import pandas as pd
import torch
import torch.utils.data as data_utils
from src.model.neural_loader import NeuralLoader
from src.model.device_data import DeviceData

class PreLoading:
    def classDistribution(self, train_data) -> any:
        label_desc = {i : x for i, x in enumerate(list(train_data.classes))}
        label_desc_inv = {v : k for k, v in label_desc.items()}
        print(label_desc_inv)
        return label_desc

    def dataInstances(self, train_data, test_data):
        print("Train instances: {} ({})".format(
            len(train_data),
            pd.Series([y for _, y in train_data]).value_counts().to_dict()
        ))

        print("Test instances: {} ({})".format(
            len(test_data),
            pd.Series([y for _, y in test_data]).value_counts().to_dict()
        ))
    
    def dataLoaders(self, batch_size, train_data, test_data) -> any:
        train_loader = data_utils.DataLoader(train_data, batch_size=batch_size, shuffle=True)
        test_loader = data_utils.DataLoader(test_data, batch_size=batch_size, shuffle=False)

        train_iter = iter(train_loader)
        train_images, train_labels = next(train_iter)

        test_iter = iter(test_loader)
        test_images, test_labels = next(test_iter)

        print('Shape de images:', train_images.shape)
        print('Shape de labels:', train_labels.shape)

        return NeuralLoader(train_loader, train_images, train_labels, test_loader, test_images, test_labels)

    def selectDevice(self) -> DeviceData:
        train_on_gpu = torch.cuda.is_available()
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

        return DeviceData(train_on_gpu, device)
