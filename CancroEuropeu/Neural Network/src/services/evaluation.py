import numpy as np
import pandas as pd
import torch
import os
import math
from array import array
from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score
from sklearn.metrics import f1_score


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
            'f1': f1_score(y_true, y_pred, pos_label=pos_label)
        }

    def predict(self, model, loader, device: str) -> any:
        y_true = []
        y_pred = []
        outputs_0 = []
        outputs_1 = []
        y_true_output = []
        class_0 = []
        class_1 = []

        for X, y in loader:
            X, y = X.to(device), y.to(device)

            output = model(X)

            _, y_pred_ = torch.max(output, 1)
            for y_ in y.cpu():
                y_true.append(y_)
                y_true_output.append(y_.item())
            for y_ in y_pred_.cpu():
                y_pred.append(y_)
            for y_ in output:
                outputs_0.append(y_.detach().numpy()[0])
                outputs_1.append(y_.detach().numpy()[1])
                softmax_0, softmax_1 = self.calculate_softmax(y_.detach().numpy()[0], y_.detach().numpy()[1])
                class_0.append(softmax_0)
                class_1.append(softmax_1)

        output_stacked = np.stack((outputs_0, outputs_1, y_true_output, class_0, class_1), axis=1)

        return y_true, y_pred, output_stacked

    def calculate_result(self, model_ft, test_loader, model_name, device: str):
        y_true, y_pred, output_stacked = self.predict(model_ft, test_loader, device)
        h_val = self.evaluate_model(y_true, y_pred)
        h_val['model_name'] = model_name
        self.h_list_val.append(h_val)

        h_list_df = pd.DataFrame(output_stacked)
        os.makedirs('output/result/output_data', exist_ok=True)
        h_list_df.to_csv(f'output/result/output_data/{model_name}.csv', index=False, sep=';',
                         header=["outputs_0", "outputs_1", "true_values", "class_0", "class_1"], decimal=",")

    def show_result(self):
        h_list_df = pd.DataFrame(self.h_list_val)
        os.makedirs('output/result', exist_ok=True)
        h_list_df.to_csv('output/result/result.csv', index=False, sep=';', decimal=",")
        h_list_df

    def calculate_softmax(self, output1, output2) -> tuple[float, float]:
        class_0 = math.exp(output1) / (math.exp(output1) + math.exp(output2))
        class_1 = math.exp(output2) / (math.exp(output1) + math.exp(output2))
        return class_0, class_1
