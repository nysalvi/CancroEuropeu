from torch.utils.tensorboard import SummaryWriter
from sklearn.metrics import mean_absolute_error
from sklearn.metrics import confusion_matrix
from sklearn.metrics import precision_score
from sklearn.metrics import accuracy_score
from sklearn.metrics import recall_score
from sklearn.metrics import f1_score
from ..utils.global_info import Info
from array import array
import pandas as pd
import numpy as np
import torch
import math
import csv
import os

class Evaluation:    
    h_list_val: array
    def __init__(self) -> None:        
        self.h_list_val = []
        pass

    def evaluate_model(self, y_true, y_pred, pos_label=1) -> any:
        return {
            'acc': accuracy_score(y_true, y_pred),
            'confusion_matrix': confusion_matrix(y_true, y_pred),
            'prec': precision_score(y_true, y_pred, pos_label=pos_label),
            'recall': recall_score(y_true, y_pred, pos_label=pos_label),
            'f1': f1_score(y_true, y_pred, pos_label=pos_label),
            'loss': mean_absolute_error(y_true, y_pred)
        }

    def predict(self, model, loader) -> any:
        y_true = []
        y_pred = []
        outputs_0 = []
        outputs_1 = []
        y_true_output = []
        class_0 = []
        class_1 = []

        for X, y in loader:
            X, y = X.to(Info.Device), y.to(Info.Device)

            output = model(X)

            _, y_pred_ = torch.max(output, 1)
            for y_ in y.cpu():
                y_true.append(y_)
                y_true_output.append(y_.item())
            for y_ in y_pred_.cpu():
                y_pred.append(y_)
            for y_ in output:
                outputs_0.append(y_.detach().cpu().numpy()[0])
                outputs_1.append(y_.detach().cpu().numpy()[1])
                softmax_0, softmax_1 = self.calculate_softmax(y_.detach().cpu().numpy()[0], y_.detach().cpu().numpy()[1])
                class_0.append(softmax_0)
                class_1.append(softmax_1)

        output_stacked = np.stack((outputs_0, outputs_1, y_true_output, class_0, class_1), axis=1)

        return y_true, y_pred, output_stacked

    def calculate_result(self, model_ft, test_loader):
        y_true, y_pred, output_stacked = self.predict(model_ft, test_loader)
        h_list_df = pd.DataFrame(output_stacked)        
        h_list_df.to_csv(f'{Info.PATH}/{Info.Name}_{Info.Optim}.csv', 
            index=False, sep=';', header=["outputs_0", "outputs_1", "true_values", "class_0", "class_1"], decimal=",")

        h_val = self.evaluate_model(y_true, y_pred)
        h_val['model_name'] = Info.Name
        h_val['LR'] = Info.LR
        h_val['momen'] = Info.Momentum
        h_val['epochs'] = Info.Epoch
        #h_val['metric'] = Info.SaveType 
        #h_val['LR'] = Info.WeightDecay
        #h_val['lr_decay'] = Info.LR_Decay                
        self.h_list_val.append(h_val)        
        self.append_results()

        Info.Writer.add_scalar(f"{Info.BoardX}/Test/Loss", h_val['loss'])
        Info.Writer.add_scalar(f"{Info.BoardX}/Test/Accuracy", h_val['acc'])
        Info.Writer.add_scalar(f"{Info.BoardX}/Test/F1", h_val['f1'])
        Info.Writer.add_scalar(f"{Info.BoardX}/Test/Precision", h_val['prec'])
        Info.Writer.add_scalar(f"{Info.BoardX}/Test/Recall", h_val['recall'])
        Info.Writer.flush()

    def append_results(self):
        h_list_df = pd.DataFrame(self.h_list_val)        
        exists = not os.path.exists('./output/results/all_results.csv')        
        h_list_df.to_csv('D:output/results/all_results.csv', mode='a', header=exists, index=False, sep=';', decimal=",")        

    def calculate_softmax(self, output1, output2) -> tuple[float, float]:
        class_0 = math.exp(output1) / (math.exp(output1) + math.exp(output2))
        class_1 = math.exp(output2) / (math.exp(output1) + math.exp(output2))
        return class_0, class_1

    @staticmethod
    def best_results():                
        df = pd.read_csv('./output/results/all_results.csv', mode='r', sep=';', decimal=',')
        idx = df.groupby('model_name')['acc'].idxmax()
        best_df = df.iloc[idx]
        best_df.to_csv('./output/results/best_results.csv', index=False, header=True, sep=';', decimal=",")
        
