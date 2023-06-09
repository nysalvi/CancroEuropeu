from sklearn.metrics import precision_recall_fscore_support, accuracy_score
from ..utils.global_info import Info
from sklearn import metrics
import pandas as pd
import csv
import os

class Analysis:
    @staticmethod
    def run() -> any:
        df = pd.read_csv(f'{Info.PATH}{os.sep}result.csv', sep=";", decimal=",")
        y = df.true_values.values
        scores = df.scores.values
        
        precision, recall, thresholds = metrics.precision_recall_curve(y, scores, pos_label=0)
        display = metrics.PrecisionRecallDisplay(precision=precision, recall=recall)
        display.plot()                
        display.figure_.savefig(f"{Info.PATH}{os.sep}prec_recall.png")

        best_thresholds = [(scores[i] + scores[i + 1]) / 2 for i, yi in enumerate(y[:-1]) if y[i] != y[i + 1]]
        results = []
        for t in best_thresholds:
            y_pred = [0 if score >= t else 1 for score in scores]
            accuracy = accuracy_score(y, y_pred)
            report = precision_recall_fscore_support(y, y_pred)
            report_b05 = precision_recall_fscore_support(y, y_pred, beta=0.5)
            report_b2 = precision_recall_fscore_support(y, y_pred, beta=2)
            results.append({   
                "model_name": Info.Name,
                "threshold": t,
                "Accuracy" : accuracy,
                "precision_0": report[0][0],
                "precision_0_05": report_b05[0][0],
                "precision_0_2": report_b2[0][0],
                "recall_0": report[1][0],
                "recall_0_05": report_b05[1][0],
                "recall_0_2": report_b2[1][0],
                "fscore_0": report[2][0],
                "fscore_0_05": report_b05[2][0],
                "fscore_0_2": report_b2[2][0]
            })
        df_results = pd.DataFrame(results)        
        exists = not os.path.exists('D:output{os.sep}results{os.sep}all_thresolds.csv')        
        
        df_results.to_csv(f'{Info.PATH}{os.sep}automatic_results.csv', index=False, sep=';', header=True, decimal=",")
        
        df_results.to_csv('D:output{os.sep}results{os.sep}all_thresolds.csv', mode='a', header=exists, index=False, sep=';', decimal=",")        

        fpr, tpr, thresholds = metrics.roc_curve(y, scores, pos_label=0)
        roc_auc = metrics.auc(fpr, tpr)
        display = metrics.RocCurveDisplay(fpr=fpr, tpr=tpr, roc_auc=roc_auc,
                                          estimator_name='estimator')
        display.plot()                
        display.figure_.savefig(f"{Info.PATH}{os.sep}estimated_positives.png")

    @staticmethod
    def best_thresolds():
        df = pd.read_csv('D:output{os.sep}results{os.sep}all_thresolds.csv')    
        if Info.SaveType == 'FScore':
            metric = 'fscore_0_05'
        elif Info.Savetype == 'Accuracy': 
            metric = 'Accuracy'

        idx = df.groupby('model_name')[metric].idxmax()
        best_df = df.iloc[idx]
        best_df.to_csv('D:output{os.sep}results{os.sep}best_thresolds.csv', index=False, header=True, sep=';', decimal=",")
