from torch.utils.tensorboard import SummaryWriter
from sklearn.metrics import mean_absolute_error
from sklearn.metrics import confusion_matrix
from sklearn.metrics import precision_score
from sklearn.metrics import accuracy_score
from sklearn.metrics import recall_score
from sklearn.metrics import f1_score, fbeta_score
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
            'fbeta': fbeta_score(y_true, y_pred, beta=0.5, pos_label=pos_label),
            'loss': mean_absolute_error(y_true, y_pred)
        }

    def predict(self, model, loader) -> any:
        y_true = []
        y_pred = []
        outputs = []
        scores = []        

        for X, y in loader:
            X, y = X.to(Info.Device), y.to(Info.Device)
            score = model(X).squeeze()                                    
            scores+= score.tolist()

            outputs = (score>=0.5).float()
            y_pred += outputs.tolist()
            y_true += y.tolist()        
        stack = np.stack((y_pred, y_true, scores), axis=1)
        return y_true, y_pred, stack

    def calculate_result(self, model_ft, test_loader, epochs):
        y_true, y_pred, stack = self.predict(model_ft, test_loader)
        h_list_df = pd.DataFrame(stack)       
        print(h_list_df)
        h_list_df.to_csv(f'{Info.PATH}/result.csv', 
            index=False, sep=';', header=["outputs", "true_values", "scores"], decimal=",")

        h_val = self.evaluate_model(y_true, y_pred)
        h_val['model_name'] = Info.Name
        h_val['LR'] = Info.LR
        h_val['momen'] = Info.Momentum
        h_val['Epochs'] = epochs
        h_val['metric'] = Info.SaveType 
        h_val['Weight_Decay'] = Info.WeightDecay         
        h_val['lr_decay'] = Info.LR_Decay                

        self.h_list_val.append(h_val)        
        self.append_results()

        #Info.Writer.add_scalar(f"{Info.Name}/Test/Loss", h_val['loss'])
        #Info.Writer.add_scalar(f"{Info.Name}/Test/Accuracy", h_val['acc'])
        #Info.Writer.add_scalar(f"{Info.Name}/Test/F1", h_val['f1'])
        #Info.Writer.add_scalar(f"{Info.Name}/Test/Precision", h_val['prec'])
        #Info.Writer.add_scalar(f"{Info.Name}/Test/Recall", h_val['recall'])
        #Info.Writer.flush()

    def append_results(self):
        h_list_df = pd.DataFrame(self.h_list_val)      
        os.makedirs('D:output/results', exist_ok=True)
        exists = not os.path.exists('D:output/results/all_results.csv')        
        h_list_df.to_csv('D:output/results/all_results.csv', mode='a', header=exists, index=False, sep=';', decimal=",")        

    @staticmethod
    def best_results():                
        df = pd.read_csv('D:output/results/all_results.csv', sep=';', decimal=',')
        idx = df.groupby('model_name')['fbeta'].idxmax()
        best_df = df.iloc[idx]
        best_df.to_csv('D:output/results/best_results.csv', index=False, header=True, sep=';', decimal=",")
        
