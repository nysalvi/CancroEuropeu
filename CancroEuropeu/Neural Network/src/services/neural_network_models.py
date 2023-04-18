import torch
from torchvision import models
from torch import nn, optim
from src.model.enums.model_name_enum import ModelName
from src.model.neural_loader import NeuralLoader
from src.model.device_data import DeviceData
from src.model.model_fit_data import ModelFitData

class NeuralNetworkModels:
    def set_parameter_requires_grad(self, model, feature_extracting):
        if feature_extracting:
            for param in model.parameters():
                param.requires_grad = False
    
    def initialize_model(self, model_name: ModelName, num_classes, feature_extract, use_pretrained=True):
        model_ft = None
        input_size = 0

        if model_name == ModelName.RESNET:
            """ Resnet18
            """
            model_ft = models.resnet18(pretrained=use_pretrained)
            self.set_parameter_requires_grad(model_ft, feature_extract)
            num_ftrs = model_ft.fc.in_features
            model_ft.fc = nn.Linear(num_ftrs, num_classes)
            input_size = 224

        elif model_name == ModelName.ALEXNET:
            """ Alexnet
            """
            model_ft = models.alexnet(pretrained=use_pretrained)
            self.set_parameter_requires_grad(model_ft, feature_extract)
            num_ftrs = model_ft.classifier[6].in_features
            model_ft.classifier[6] = nn.Linear(num_ftrs,num_classes)
            input_size = 224

        elif model_name == ModelName.VGG16:
            """ VGG16
            """
            model_ft = models.vgg16(pretrained=use_pretrained)
            self.set_parameter_requires_grad(model_ft, feature_extract)
            num_ftrs = model_ft.classifier[6].in_features
            model_ft.classifier[6] = nn.Linear(num_ftrs,num_classes)
            input_size = 224

        elif model_name == ModelName.MOBILENET_V2:
            """ 
            """
            model_ft = models.mobilenet_v2(pretrained=use_pretrained)
            self.set_parameter_requires_grad(model_ft, feature_extract)
            num_ftrs = model_ft.classifier[1].in_features
            model_ft.classifier[1] = nn.Linear(num_ftrs,num_classes)
            input_size = 224

        elif model_name == ModelName.VGG:
            """ VGG11_bn
            """
            model_ft = models.vgg11_bn(pretrained=use_pretrained)
            self.set_parameter_requires_grad(model_ft, feature_extract)
            num_ftrs = model_ft.classifier[6].in_features
            model_ft.classifier[6] = nn.Linear(num_ftrs,num_classes)
            input_size = 224

        elif model_name == ModelName.MOBILENET_V3_SMALL:
            model_ft = models.mobilenet_v3_small(pretrained=use_pretrained)
            self.set_parameter_requires_grad(model_ft, feature_extract)
            num_ftrs = model_ft.classifier[3].in_features
            model_ft.classifier[3] = nn.Linear(num_ftrs,num_classes)
            input_size = 224

        elif model_name == ModelName.MOBILENET_V3_LARGE:
            model_ft = models.mobilenet_v3_large(pretrained=use_pretrained)
            self.set_parameter_requires_grad(model_ft, feature_extract)
            num_ftrs = model_ft.classifier[3].in_features
            model_ft.classifier[3] = nn.Linear(num_ftrs,num_classes)
            input_size = 224     

        elif model_name == ModelName.SQUEEZENET:
            """ Squeezenet
            """
            model_ft = models.squeezenet1_0(pretrained=use_pretrained)
            self.set_parameter_requires_grad(model_ft, feature_extract)
            model_ft.classifier[1] = nn.Conv2d(512, num_classes, kernel_size=(1,1), stride=(1,1))
            model_ft.num_classes = num_classes
            input_size = 224

        elif model_name == ModelName.DENSENET:
            """ Densenet
            """
            model_ft = models.densenet121(pretrained=use_pretrained)
            self.set_parameter_requires_grad(model_ft, feature_extract)
            num_ftrs = model_ft.classifier.in_features
            model_ft.classifier = nn.Linear(num_ftrs, num_classes)
            input_size = 224

        else:
            print("Invalid model name, exiting...")
            exit()

        return model_ft, input_size

    def optimize_model(self, device, feature_extract, model_ft) -> ModelFitData:
        model_ft = model_ft.to(device)

        params_to_update = model_ft.parameters()
        print("Params to learn:")
        if feature_extract:
            params_to_update = []
            for name,param in model_ft.named_parameters():
                if param.requires_grad == True:
                    params_to_update.append(param)
                    print("\t",name)
        else:
            for name,param in model_ft.named_parameters():
                if param.requires_grad == True:
                    print("\t",name)
        
        optimizer_ft = optim.SGD(params_to_update, lr=0.001, momentum=0.9)
        return ModelFitData(optimizer_ft, model_ft)

    def verify_predict_before_training(self, deviceData: DeviceData, neuralLoader: NeuralLoader, modelFitData: ModelFitData):
        if deviceData.isCuda: 
            _, y_pred = torch.max(modelFitData.model_ft(neuralLoader.train_loader.images.cuda()),1)
        else:
            _, y_pred = torch.max(modelFitData.model_ft(neuralLoader.train_loader.images),1)

        print("Ground turth: ", neuralLoader.train_loader.labels)
        print("Predicted   : ", y_pred)