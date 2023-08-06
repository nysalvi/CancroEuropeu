from torchvision import models
from torch import nn, optim
from ..model.enums.model_name_enum import ModelName
from ..model.neural_loader import NeuralLoader
from ..model.device_data import DeviceData
from ..model.model_fit_data import ModelFitData
from ..utils.global_info import Info
import torch
import os

class NeuralNetworkModels:
    def __init__(self):
        pass

    def set_parameter_requires_grad(self, model, feature_extracting):
        if feature_extracting:
            for param in model.parameters():
                param.requires_grad = False
    
    def initialize_model(self, model_name: ModelName, num_classes:int, feature_extract:bool, use_pretrained:bool):
        model_ft = None
        input_size = 224

        if model_name == ModelName.RESNET:
            model_ft = models.resnet18(pretrained=not use_pretrained)                        
            self.set_parameter_requires_grad(model_ft, feature_extract)
            num_ftrs = model_ft.fc.in_features            
            model_ft.fc = nn.Linear(num_ftrs, num_classes)                                    
            grad_layer = model_ft.layer4[-1]
            
        #elif model_name == ModelName.ALEXNET:
        #    model_ft = models.alexnet(pretrained=not use_pretrained)            
        #    self.set_parameter_requires_grad(model_ft, feature_extract)
        #    num_ftrs = model_ft.classifier[6].in_features
        #    model_ft.classifier[6] = nn.Linear(num_ftrs,num_classes)

        elif model_name == ModelName.VGG16:
            model_ft = models.vgg16_bn(pretrained=not use_pretrained)            
            self.set_parameter_requires_grad(model_ft, feature_extract)
            num_ftrs = model_ft.classifier[6].in_features
            model_ft.classifier[6] = nn.Linear(num_ftrs,num_classes)            
            grad_layer = model_ft.features[-1]            
        #elif model_name == ModelName.MOBILENET_V2:
        #    model_ft = models.mobilenet_v2(pretrained=not use_pretrained)
        #    self.set_parameter_requires_grad(model_ft, feature_extract)            
        #    num_ftrs = model_ft.classifier[1].in_features
        #    model_ft.classifier[1] = nn.Linear(num_ftrs,num_classes)
        #
        #elif model_name == ModelName.VGG11:
        #    model_ft = models.vgg11_bn(pretrained=not use_pretrained)
        #    
        #    self.set_parameter_requires_grad(model_ft, feature_extract)
        #    num_ftrs = model_ft.classifier[6].in_features
        #    model_ft.classifier[6] = nn.Linear(num_ftrs,num_classes)
        #
        #elif model_name == ModelName.MOBILENET_V3_SMALL:
        #    model_ft = models.mobilenet_v3_small(pretrained=not use_pretrained)
        #    
        #    self.set_parameter_requires_grad(model_ft, feature_extract)
        #    num_ftrs = model_ft.classifier[3].in_features
        #    model_ft.classifier[3] = nn.Linear(num_ftrs,num_classes)
        #
        #elif model_name == ModelName.MOBILENET_V3_LARGE:
        #    model_ft = models.mobilenet_v3_large(pretrained=not use_pretrained)
        #
        #    self.set_parameter_requires_grad(model_ft, feature_extract)
        #    num_ftrs = model_ft.classifier[3].in_features
        #    model_ft.classifier[3] = nn.Linear(num_ftrs,num_classes)

        #elif model_name == ModelName.SQUEEZENET:
        #    model_ft = models.squeezenet1_0(pretrained=not use_pretrained)
        #
        #    self.set_parameter_requires_grad(model_ft, feature_extract)
        #    model_ft.classifier[1] = nn.Conv2d(512, num_classes, kernel_size=(1,1), stride=(1,1))
        #    model_ft.num_classes = num_classes        
            
        #elif model_name == ModelName.DENSENET:
        #    model_ft = models.densenet121(pretrained= not use_pretrained)             
        #    
        #    model_ft.load_state_dict(torch.load('./data/densenet.pth'))
        #    self.set_parameter_requires_grad(model_ft, feature_extract)
        #    num_ftrs = model_ft.classifier[1].in_features
        #    model_ft.classifier[1] = nn.Linear(num_ftrs, num_classes)
        #    grad_layer = model_ft.features[-1]

        elif model_name == ModelName.VGG19:
            model_ft = models.vgg19_bn(pretrained=not use_pretrained)                   
            num_ftrs = model_ft.classifier[6].in_features
            self.set_parameter_requires_grad(model_ft, feature_extract)            
            model_ft.classifier[6] = nn.Linear(num_ftrs, num_classes)  
            grad_layer = model_ft.features[-1]

        elif model_name == ModelName.RESNET50:
            model_ft = models.resnet50(pretrained=not use_pretrained)               
            self.set_parameter_requires_grad(model_ft, feature_extract)
            num_ftrs = model_ft.fc.in_features
            model_ft.fc = nn.Linear(num_ftrs, num_classes)
            grad_layer = model_ft.layer4[-1]

        elif model_name == ModelName.INCEPTIONV3:
            model_ft = models.inception.inception_v3(pretrained=not use_pretrained)   
            self.set_parameter_requires_grad(model_ft, feature_extract)
            num_ftrs = model_ft.fc.in_features            
            model_ft.fc = nn.Linear(num_ftrs, num_classes)            
            grad_layer = model_ft.Mixed_7c.branch_pool
            input_size = 299          
        else:
            print("Invalid model name, exiting...")
            exit()
        if use_pretrained:
            model_ft.load_state_dict(torch.load(f'{Info.PATH}{os.sep}state_dict.pt'))
        return model_ft, input_size, grad_layer

    def optimize_model(self, feature_extract, model_ft, optimizer=False, 
                    lr:float=0.001, momentum:float=0.9) -> ModelFitData:
        model_ft = model_ft.to(Info.Device)

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
        optimizer_ft = eval(optimizer) if optimizer else optim.SGD(params_to_update, lr=lr, momentum=momentum, 
                                                            weight_decay=Info.WeightDecay)
        
        return ModelFitData(optimizer_ft, model_ft)

    def verify_predict_before_training(self, deviceData: DeviceData, neuralLoader: NeuralLoader, modelFitData: ModelFitData):
        modelFitData.model_ft.eval()
        if deviceData.isCuda: 
            _, y_pred = torch.max(modelFitData.model_ft(neuralLoader.train_loader.images.cuda()),1)
        else:
            _, y_pred = torch.max(modelFitData.model_ft(neuralLoader.train_loader.images),1)

        print("Ground turth: ", neuralLoader.train_loader.labels)
        print("Predicted   : ", y_pred)                
        