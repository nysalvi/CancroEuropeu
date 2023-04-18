import pandas as pd
from sklearn import metrics
import os
from sklearn.metrics import precision_recall_fscore_support

class Analysis:

    def run(self, model_name) -> any:
        df = pd.read_csv(f"output/result/output_data/{model_name}.csv", sep=";", decimal=",")
        y = df.true_values.values
        scores = -df.outputs_1.values
        precision, recall, thresholds = metrics.precision_recall_curve(y, scores, pos_label=0)
        display = metrics.PrecisionRecallDisplay(precision=precision, recall=recall)
        display.plot()
        os.makedirs('output/images/analysis/prec_recall', exist_ok=True)
        display.figure_.savefig(f"./output/images/analysis/prec_recall/{model_name}.png")

        best_thresholds = [(scores[i] + scores[i + 1]) / 2 for i, yi in enumerate(y[:-1]) if y[i] != y[i + 1]]
        results = []
        for t in best_thresholds:
            y_pred = [0 if score >= t else 1 for score in scores]
            report = precision_recall_fscore_support(y, y_pred)
            report_b05 = precision_recall_fscore_support(y, y_pred, beta=0.5)
            report_b2 = precision_recall_fscore_support(y, y_pred, beta=2)
            results.append({
                "threshold": t,
                "precision_0": report[0][0],
                "recall_0": report[1][0],
                "fscore_0": report[2][0],
                "fscore_0_05": report_b05[2][0],
                "fscore_0_2": report_b2[2][0]
            })
        df_results = pd.DataFrame(results)
        os.makedirs('output/result/output_data/automatic_results', exist_ok=True)
        df_results.to_csv(f'output/result/output_data/automatic_results/{model_name}.csv', index=False, sep=';', header=True, decimal=",")

        fpr, tpr, thresholds = metrics.roc_curve(y, scores, pos_label=0)
        roc_auc = metrics.auc(fpr, tpr)
        display = metrics.RocCurveDisplay(fpr=fpr, tpr=tpr, roc_auc=roc_auc,
                                          estimator_name='estimator')
        display.plot()
        os.makedirs('output/images/analysis/estimated_positives', exist_ok=True)
        display.figure_.savefig(f"./output/images/analysis/estimated_positives/{model_name}.png")