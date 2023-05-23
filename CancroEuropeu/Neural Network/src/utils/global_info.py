from torch.utils.tensorboard import SummaryWriter
from ..model.enums.model_name_enum import ModelName
import argparse
import os

class Info():    
    default = {}    
    Writer = None
    Device = ''
    Name = ''
    Optim = ''
    LR = 0
    LR_Decay = 0
    WeightDecay = 0
    Momentum = 0
    SaveType = ''
    DataPath = ''    
    PATH = ''
    BoardPATH = ''
    BoardX = ''    
    Grad = []
    Activation = []

    @staticmethod
    def args():
        parser = argparse.ArgumentParser()        
        #choices=['ALL', 'RESNET', 'ALEXNET', 'VGG16', 'MOBILENET_V2', 'VGG', 'MOBILENET_V3_SMALL', 'MOBILENET_V3_LARGE', 'SQUEEZENET', 'DENSENET']
        parser.add_argument('--Name', type=str, default='ALL', help='choose whether to train with a single model or all of them')        
        parser.add_argument('--num_epochs', type=int, default=100, help='max number of training epochs to execute')
        parser.add_argument('--num_classes', type=int, default=1, help='TOTAL number of categories images can be classified. Does not determinate if image can be multi-label classified')
        parser.add_argument('--batch_size', type=int, default=8, help='number of images per batch; aka mini-batch size')
        parser.add_argument('--no_pre_trained', default=True, action='store_false', help='set the model as NOT pre-trained')
        parser.add_argument('--no_feature_extract', default=True, action='store_false', help='set the model for NO features extraction')
        parser.add_argument('--Device', type=str, default='cuda:0', choices= ['cpu', 'cuda', f'cuda:{int}'], help='use "cpu", "cuda" or "cuda:x"; where "x" is gpu number, if "x" is omitted default value is 0')
        parser.add_argument('--Optim', default='SGD', choices=['SGD', ''], help='list of optimizers available')
        parser.add_argument('--LR', type=float, default=0.0001, help='initial learning rate value')
        parser.add_argument('--LR_Decay', type=float, default=1, help='learning rate decayment')
        parser.add_argument('--WeightDecay', type=float, default=0.01, help='decayment of parameters weight')            
        parser.add_argument('--Momentum', type=float, default=0.9, help='momentum for optimizer')
        parser.add_argument('--SaveType', type=str, default='Accuracy', choices=['Accuracy', 'FScore'], help='save model based on characteristic')        
        parser.add_argument('--PATH', type=str, default=False, help='base path for saving model')        
        parser.add_argument('--BoardPATH', type=str, default=False, help='base path for saving tensorboard event files')        
        parser.add_argument('--DataPath', type=str, default=f'{os.path.join(os.getcwd(), "data")}', help='path for where the data is')
        parser.add_argument('--image_size', type=int, default=224, help='size of the input images; It is expected that the dimensions are is squared.')
        
        args = vars(parser.parse_args())          
        constants = {}
        for key, value in args.items() : 
            Info.default.update({key : parser.get_default(key)})                        
            try:
                getattr(Info, key)
                setattr(Info, key, value)                
            except AttributeError:
                constants.update({key: value})                                                
        return constants

    @staticmethod
    def update_path(name):
        Info.Name = name        
        Info.PATH = f"D:/output/{Info.Name}_{Info.SaveType}/{Info.Optim}_{Info.Momentum}/LR_{Info.LR}_Decay_{Info.LR_Decay}_Weight_{Info.WeightDecay}"
        Info.BoardX = f"D:/output/runs/LR_{Info.LR}_{Info.LR_Decay}_Weight_{Info.WeightDecay}/{Info.Optim}_{Info.Momentum}_{Info.SaveType}/"
        Info.Writer = SummaryWriter(Info.BoardX)