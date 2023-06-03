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
import torch
import gc
import os

warnings.filterwarnings('ignore')
random.RandomState(1)

#constants
constants = Info.args()
batch_size = constants['batch_size']
num_classes = constants['num_classes']
feature_extract = constants['no_feature_extract']
use_pretrained = constants['no_pre_trained']
num_epochs = constants['num_epochs']

#End constants
selected_models = Models.select_models(ModelName[Info.Name])
#Images.download(currentTempFolder, "data.zip")
#Images.unzip(currentTempFolder, "data.zip")g


evaluation = Evaluation()
    
for neural_model_name in selected_models:    
    Info.update_path(neural_model_name.name)

    print(f"\n-------------------- Current model: {neural_model_name.name} --------------------\n")
    neuralNetworkModels = NeuralNetworkModels()
    model_ft, input_size, grad_layer = neuralNetworkModels.initialize_model(neural_model_name, num_classes, feature_extract, use_pretrained = use_pretrained)

    preProcessing = PreProcessing(Info.DataPath, input_size, input_size)
    neuralData = preProcessing.run()
    preLoading = PreLoading()
    label_desc = preLoading.classDistribution(neuralData.train_data)
    device = preLoading.selectDevice()    
    neuralLoader = preLoading.dataLoaders(batch_size, neuralData.train_data, neuralData.dev_data, neuralData.test_data)

    modelFitData = neuralNetworkModels.optimize_model(feature_extract, model_ft, lr=Info.LR, momentum=Info.Momentum)    
    training = Training(torch.optim.lr_scheduler.CosineAnnealingLR,  modelFitData.optimizer_ft, num_epochs)
    criterion = nn.BCEWithLogitsLoss().cuda() if device.isCuda else nn.BCEWithLogitsLoss()
    dataframe, model_ft = training.train_and_evaluate(model_ft, neuralLoader.train_loader.loader, neuralLoader.dev_loader.loader, criterion)      
    evaluation.calculate_result(model_ft, neuralLoader.test_loader.loader, num_epochs)    
    #training.verify_images(neuralLoader.test_loader.loader, batch_size, model_ft, label_desc, grad_layer)    
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