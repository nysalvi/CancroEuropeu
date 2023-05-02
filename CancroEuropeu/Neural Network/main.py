from src.services.neural_network_models import NeuralNetworkModels
from src.services.pre_processing import PreProcessing
from src.model.enums.model_name_enum import ModelName
from src.services.pre_loading import PreLoading
from src.services.evaluation import Evaluation
from src.services.training import Training
from src.services.analysis import Analysis
from src.utils.global_info import Info
from .src.utils.folder import Folder
#from src.utils.images import Images
from src.utils.models import Models
from numpy import random
from torch import nn
import warnings
import os

warnings.filterwarnings('ignore')
random.RandomState(1)

#constants
constants = Info.args()
batch_size = constants['batch_size']
num_classes = constants['num_classes']
image_size = constants['image_size']
feature_extract = constants['no-feature_extract']
use_pretrained = constants['no-pre_trained']
num_epochs = constants['num_epochs']

#End constants

models = Models.select_models(Info.Name)

#Images.download(currentTempFolder, "data.zip")
#Images.unzip(currentTempFolder, "data.zip")

preProcessing = PreProcessing(Info.DataPath, (image_size, image_size))
neuralData = preProcessing.run()

preLoading = PreLoading()
label_desc = preLoading.classDistribution(neuralData.train_data)
preLoading.dataInstances(neuralData.train_data, neuralData.dev_data, neuralData.test_data)
neuralLoader = preLoading.dataLoaders(batch_size, neuralData.train_data, neuralData.dev_data, neuralData.test_data)
device = preLoading.selectDevice()

evaluation = Evaluation()
os.makedirs(f'{Info.PATH}', exist_ok=True)
for neural_model_name in models:    
    Info.update_path(neural_model_name.name)

    print(f"\n-------------------- Current model: {neural_model_name.name} --------------------\n")
    neuralNetworkModels = NeuralNetworkModels()
    model_ft, input_size = neuralNetworkModels.initialize_model(neural_model_name, num_classes, feature_extract, use_pretrained = use_pretrained)
    modelFitData = neuralNetworkModels.optimize_model(feature_extract, model_ft, lr=Info.LR, momentum=Info.Momentum)
    neuralNetworkModels.verify_predict_before_training(device, neuralLoader, modelFitData)

    training = Training()
    criterion = nn.CrossEntropyLoss().cuda() if device.isCuda else nn.CrossEntropyLoss()
    dataframe, model_ft = training.train_and_evaluate(model_ft, num_epochs, neuralLoader.train_loader.loader, neuralLoader.dev_loader.loader, modelFitData.optimizer_ft, criterion)      

    #torch.save(model_ft, f"{Info.PATH}/{Info.Name}_{Info.Optim}_model.pth")

    training.verify_images(neuralLoader.test_loader.loader, batch_size, model_ft, label_desc)
    
    evaluation.calculate_result(model_ft, neuralLoader.test_loader.loader)    
    Analysis.run()   