#!/usr/bin/env python

from src.services.neural_network_models import NeuralNetworkModels
from src.services.pre_processing import PreProcessing
from src.model.enums.model_name_enum import ModelName
from src.services.pre_loading import PreLoading
from src.services.evaluation import Evaluation
from src.services.training import Training
from src.services.analysis import Analysis
from src.utils.global_info import Info
#from src.utils.folder import Folder
#from src.utils.images import Images
from src.utils.models import Models
from torchvision import models
from numpy import random
from torch import nn
import warnings
import pickle
import binascii
import torch
import gc, json
import os
import io

warnings.filterwarnings('ignore')
random.RandomState(1)
#constants
constants = Info.args()
batch_size = constants['batch_size']
num_classes = constants['num_classes']
feature_extract = constants['no_feature_extract']

#End constants
selected_models = Models.select_models(ModelName[Info.Name])
#Images.download(currentTempFolder, "data.zip")
#Images.unzip(currentTempFolder, "data.zip")g


evaluation = Evaluation()

remove = ['args', 'update_path', 'info_list', 'Writer', 'Activation', 'default']
Info.info_list = [x for x in dir(Info) if not x.startswith('_')]    
for x in remove: Info.info_list.remove(x)

for neural_model_name in selected_models:    
    Info.Completed = False
    Info.update_path(neural_model_name.name)    
    
    resume_training = os.path.exists(f'{Info.PATH}{os.sep}stats.txt')

    if resume_training and os.path.getsize(f'{Info.PATH}{os.sep}stats.txt') > 0:                    
        file_ = open(f'{Info.PATH}{os.sep}stats.txt', 'r')
        json_save = file_.read()
        file_.close()                
        info_dict = json.loads(json_save)
        for x, y in info_dict.items():
            try:
                if float(y) == int(y):
                    setattr(Info, x, int(y))
                else:
                    setattr(Info, x, float(y))
            except:
                setattr(Info, x, y)
                    
        if bool(Info.Completed) or Info.Epoch == Info.Epochs: 
            continue
    os.makedirs(Info.PATH, exist_ok=True)
    os.makedirs(Info.BoardX, exist_ok=True)    
    print(f"\n-------------------- Current model: {neural_model_name.name} --------------------\n")
    neuralNetworkModels = NeuralNetworkModels()
    model_ft, input_size, grad_layer = neuralNetworkModels.initialize_model(neural_model_name, num_classes, 
        feature_extract, use_pretrained = resume_training)

    preProcessing = PreProcessing(Info.DataPath, input_size, input_size)
    neuralData = preProcessing.run()
    preLoading = PreLoading()
    label_desc = preLoading.classDistribution(neuralData.train_data)
    device = preLoading.selectDevice()    
    neuralLoader = preLoading.dataLoaders(batch_size, neuralData.train_data, neuralData.dev_data, neuralData.test_data)

    modelFitData = neuralNetworkModels.optimize_model(feature_extract, model_ft, lr=Info.LR, momentum=Info.Momentum)    
    training = Training(torch.optim.lr_scheduler.CosineAnnealingLR,  modelFitData.optimizer_ft, Info.Epochs)
    criterion = nn.BCEWithLogitsLoss().cuda() if device.isCuda else nn.BCEWithLogitsLoss()
    dataframe, model_ft = training.train_and_evaluate(model_ft, neuralLoader.train_loader.loader, 
        neuralLoader.dev_loader.loader, criterion)      
    evaluation.calculate_result(model_ft, neuralLoader.test_loader.loader, Info.Epochs)    
    #training.verify_images("last_conv", neuralLoader.test_loader.loader, batch_size, model_ft, label_desc, grad_layer)        
    Analysis.run()   
    
    #del criterion
    #del modelFitData
    #del neuralLoader
    #del neuralData
    #del training.optimizer
    #del training.scheduler
    #del model_ft
    #
    #torch.cuda.empty_cache()
    #gc.collect()
    
    
    #  74.88it/s   50m59s  # 
    #  LR_ 5e-5_           #
    #  Weight_ 1e-8_       #