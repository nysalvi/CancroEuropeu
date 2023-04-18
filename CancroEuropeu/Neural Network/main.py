from torch import nn
from src.utils.folder import Folder
from src.utils.images import Images
from src.utils.models import Models
from src.model.enums.model_name_enum import ModelName
from src.services.pre_processing import PreProcessing
from src.services.pre_loading import PreLoading
from src.services.neural_network_models import NeuralNetworkModels
from src.services.training import Training
from src.services.evaluation import Evaluation
from src.services.analysis import Analysis

import warnings
from numpy import random
warnings.filterwarnings('ignore')
random.RandomState(1)

#constants
model_name = ModelName.ALL
batch_size = 8
num_classes = 2
feature_extract = True
use_pretreined = True
num_epochs = 10
#End constants

models = Models.select_models(model_name)
Folder.createFolder(None, "temp")
Folder.createFolder(None, "output")
currentTempFolder = Folder.getCurrentFolder() + "/temp/"

Images.download(currentTempFolder, "data.zip")
Images.unzip(currentTempFolder, "data.zip")

preProcessing = PreProcessing(currentTempFolder + "/data", 224, 224)
neuralData = preProcessing.run()

preLoading = PreLoading()
label_desc = preLoading.classDistribution(neuralData.train_data)
preLoading.dataInstances(neuralData.train_data, neuralData.test_data)
neuralLoader = preLoading.dataLoaders(batch_size, neuralData.train_data, neuralData.test_data)
device = preLoading.selectDevice()

evaluation = Evaluation()
for neural_model_name in models:
    print(f"\n-------------------- Current model: {neural_model_name.name} --------------------\n")
    neuralNetworkModels = NeuralNetworkModels()
    model_ft, input_size = neuralNetworkModels.initialize_model(neural_model_name, num_classes, feature_extract, use_pretrained = use_pretreined)
    modelFitData = neuralNetworkModels.optimize_model(device.device, feature_extract, model_ft)
    neuralNetworkModels.verify_predict_before_training(device, neuralLoader, modelFitData)

    training = Training()
    criterion = nn.CrossEntropyLoss()
    dataframe, model_ft = training.train_and_evaluate(model_ft, num_epochs, neuralLoader.train_loader.loader, neuralLoader.test_loader.loader, modelFitData.optimizer_ft, criterion, device.device)  
    training.exportModel(model_ft, neural_model_name.name)
    training.verify_images(neuralLoader.test_loader.loader, batch_size, model_ft, device.device, label_desc, neural_model_name.name)
    
    evaluation.calculate_result(model_ft, neuralLoader.test_loader.loader, neural_model_name.name, device.device)

    analysis = Analysis()
    analysis.run(neural_model_name.name)

evaluation.show_result()