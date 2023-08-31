from sklearn.metrics import roc_auc_score, roc_curve, auc, fbeta_score
from sklearn.metrics import precision_score
from sklearn.metrics import accuracy_score
from sklearn.metrics import recall_score
from sklearn.metrics import f1_score, fbeta_score

from ..utils.global_info import Info
import matplotlib.pyplot as plt
from torch import nn, optim
from tqdm.auto import tqdm
import pandas as pd
import numpy as np
import binascii
import torch
import json
import math
import pickle
import time
import os
import io


class Training:   
    def __init__(self, scheduler, optimizer, num_epochs):
        self.optimizer = optimizer
        self.num_epochs = num_epochs
        self.scheduler = scheduler(optimizer, T_max=num_epochs, last_epoch=-1)
        for i in range(Info.Epoch): self.scheduler.step()
        
    def train_epoch(self, model, trainLoader, criterion):
        model.train()
        losses = []      
        total = 0
        ground_truth = np.zeros(shape=len(trainLoader.dataset))
        prediction = np.zeros(shape=len(trainLoader.dataset))
        scores = np.zeros(shape=len(trainLoader.dataset))        
        cont = 0
        correct = 0
        for X, y in trainLoader:    
            X, y = X.to(Info.Device), y.to(Info.Device)            
            ground_truth[cont:cont + len(X)] = y.cpu().data.numpy()                

            # (1) Passar os dados pela rede neural (forward)
            if model._get_name() == 'Inception3':
                output = model(X)[0].squeeze()            
            else:
                output = model(X).squeeze()                       
            probability_output = nn.Sigmoid()(output)
            scores[cont:cont + len(X)] = probability_output.cpu().data.numpy()
            y_pred = (probability_output >= 0.5).float()
            prediction[cont:cont + len(X)] = y_pred.cpu().data.numpy()                
            total += len(y)

            # (2) Calcular o erro da saída da rede com a classe das instâncias (loss)                    
            loss = criterion(output, y.float())        
            # (3) Usar o erro para calcular quanto cada peso (wi) contribuiu com esse erro (backward)
            loss.backward()
            # (4) Ataulizar os pesos da rede neural
            self.optimizer.step()        
            self.optimizer.zero_grad()

            losses.append(loss.item())
            correct += (y_pred == y).sum().cpu().data.numpy()                
            cont+=len(X)

        self.scheduler.step()

        acc = correct/total
        fbeta = fbeta_score(ground_truth, prediction, beta=0.5)                                                                            
        fscore = f1_score(ground_truth, prediction)
        precision = precision_score(ground_truth, prediction)
        recall = recall_score(ground_truth, prediction)
        fpr, tpr, _ = roc_curve(ground_truth, scores)
        roc_auc = auc(fpr, tpr)

        model.eval()

        measures = {'loss' : np.mean(losses), 'fbeta' : fbeta, 'acc': acc, 'fscore': fscore, 
                    'prec': precision, 'recall':recall, 'auc':roc_auc}
        return measures
    def eval_model(self, model, loader, criterion):        
        total = 0
        correct = 0
        losses = []
        ground_truth = np.zeros(shape=len(loader.dataset))
        prediction = np.zeros(shape=len(loader.dataset))
        scores = np.zeros(shape=len(loader.dataset))
        cont = 0
        with torch.no_grad():
            for X, y in loader:                
                X, y = X.to(Info.Device), y.to(Info.Device)      
                ground_truth[cont:cont + len(X)] = y.cpu().data.numpy()                
                output = model(X).squeeze()                                      
                probability_output = nn.Sigmoid()(output)
                scores[cont:cont + len(X)] = probability_output.cpu().data.numpy()
                y_pred = (probability_output >= 0.5).float()
                prediction[cont:cont + len(X)] = y_pred.cpu().data.numpy()                
                total += len(y)
                loss = criterion(output, y.float())
                losses.append(loss.item())             
                correct += (y_pred == y).sum().cpu().data.numpy()                
                cont+=len(X)
            acc = correct/total
            #ground_truth = [x.astype(int) for x in ground_truth]
            #prediction = [x.astype(int) for x in prediction]
            fbeta = fbeta_score(ground_truth, prediction, beta=0.5)                                                                            
            fscore = f1_score(ground_truth, prediction)
            precision = precision_score(ground_truth, prediction)
            recall = recall_score(ground_truth, prediction)
            fpr, tpr, _ = roc_curve(ground_truth, scores)
            roc_auc = auc(fpr, tpr)
            
        measures = {'loss' : np.mean(losses), 'fbeta' : fbeta, 'acc': acc, 'fscore': fscore, 
                    'prec': precision, 'recall':recall, 'auc':roc_auc}
        
        return measures

    def train_and_evaluate(self, model, train_loader, dev_loader, criterion):                
        e_measures = []        
        pbar = tqdm(range(Info.Epoch, self.num_epochs))        
        header = ['model', 'mode', 'metric', 'epoch', 'lr', 'wd', 'value'] 
        if Info.Epoch == 0:
            df = pd.DataFrame([header])
            df.to_csv(f'{Info.BoardX}\\data.csv', mode='a', index=False)
        else: 
            df = pd.DataFrame([])
        for e in pbar:
            measures_on_train = self.train_epoch(model, train_loader, criterion)            
            measures_on_dev = self.eval_model(model, dev_loader, criterion)
            measures = {'epoch': Info.Epoch, 'train_loss': measures_on_train['loss'], 'train_fbeta' : measures_on_train['fbeta'], 
                    'train_acc' : measures_on_train['acc'], 'train_fscore' : measures_on_train['fscore'], 
                    'train_prec' : measures_on_train['prec'], 'train_recall' : measures_on_train['recall'],
                    'train_auc' : measures_on_train['auc'], 
                    'dev_loss': measures_on_dev['loss'], 'dev_fbeta' : measures_on_dev['fbeta'], 
                    'dev_acc' : measures_on_dev['acc'], 'dev_fscore' : measures_on_dev['fscore'], 
                    'dev_prec' : measures_on_dev['prec'], 'dev_recall' : measures_on_dev['recall'],
                    'dev_auc' : measures_on_dev['auc']  
            }
            if Info.FBeta < measures_on_dev['fbeta'].round(4):                
                Info.CurTolerance = -1
                Info.FBeta = measures_on_dev['fbeta'].round(4)
                torch.save(model.state_dict(), f'{Info.PATH}{os.sep}state_dict.pt')                

            save = {}                                           
            for x in Info.info_list: save.update({x : getattr(Info, x)})
            file_ = open(f'{Info.PATH}{os.sep}stats.txt', 'w')
            json_save = json.dump(save, file_, skipkeys=True, ensure_ascii=False)            
            file_.close()         
            

            df.loc[len(df)] = [Info.Name, 'Train', 'FBeta', Info.Epoch, Info.LR, Info.WeightDecay, measures['train_fbeta']]
            df.loc[len(df)] = [Info.Name, 'Train', 'Loss', Info.Epoch, Info.LR, Info.WeightDecay, measures['train_loss']]
            df.loc[len(df)] = [Info.Name, 'Train', 'Accuracy', Info.Epoch, Info.LR, Info.WeightDecay, measures['train_acc']]
            df.loc[len(df)] = [Info.Name, 'Train', 'Precision', Info.Epoch, Info.LR, Info.WeightDecay, measures['train_prec']]
            df.loc[len(df)] = [Info.Name, 'Train', 'Recall', Info.Epoch, Info.LR, Info.WeightDecay, measures['train_recall']]
            df.loc[len(df)] = [Info.Name, 'Train', 'AUC', Info.Epoch, Info.LR, Info.WeightDecay, measures['train_auc']]
            df.loc[len(df)] = [Info.Name, 'Train', 'FScore', Info.Epoch, Info.LR, Info.WeightDecay, measures['train_fscore']]

            df.loc[len(df)] = [Info.Name, 'Validation', 'FBeta', Info.Epoch, Info.LR, Info.WeightDecay, measures['dev_fbeta']]
            df.loc[len(df)] = [Info.Name, 'Validation', 'Loss', Info.Epoch, Info.LR, Info.WeightDecay, measures['dev_loss']]
            df.loc[len(df)] = [Info.Name, 'Validation', 'Accuracy', Info.Epoch, Info.LR, Info.WeightDecay, measures['dev_acc']]
            df.loc[len(df)] = [Info.Name, 'Validation', 'Precision', Info.Epoch, Info.LR, Info.WeightDecay, measures['dev_prec']]
            df.loc[len(df)] = [Info.Name, 'Validation', 'Recall', Info.Epoch, Info.LR, Info.WeightDecay, measures['dev_recall']]
            df.loc[len(df)] = [Info.Name, 'Validation', 'AUC', Info.Epoch, Info.LR, Info.WeightDecay, measures['dev_auc']]
            df.loc[len(df)] = [Info.Name, 'Validation', 'FScore', Info.Epoch, Info.LR, Info.WeightDecay, measures['dev_fscore']]
            
            df.to_csv(f'{Info.BoardX}\\data.csv', mode='a', index=False, header=False)

            df = pd.DataFrame([], columns=header)

            Info.CurTolerance+= 1
            Info.Epoch += 1                

            if Info.CurTolerance == Info.Tolerance:
                break
            pbar.set_postfix(measures)     
            e_measures += [measures]
        Info.Completed = True
        save = {}                                           
        for x in Info.info_list: save.update({x : getattr(Info, x)})
        file_ = open(f'{Info.PATH}{os.sep}stats.txt', 'w')
        json_save = json.dump(save, file_, skipkeys=True, ensure_ascii=False)            
        file_.close()         

        return pd.DataFrame(e_measures), model

    def verify_images(self, name, test_loader, batch_size, model_ft, label_desc, grad_layer):               
        from pytorch_grad_cam import GradCAM
        from pytorch_grad_cam.utils.model_targets import ClassifierOutputTarget
        from pytorch_grad_cam.utils.image import show_cam_on_image
        from torchvision import transforms
        
        cam = GradCAM(model_ft, target_layers=[grad_layer], use_cuda=True)
        targets = [ClassifierOutputTarget(0)]

        rows = 2
        columns = math.ceil(batch_size / rows)

        un_divide = transforms.Normalize([0, 0, 0], [1/0.229, 1/0.224, 1/0.225])
        un_minus = transforms.Normalize([-0.485, -0.456, -0.406], [1, 1, 1])
        
        pat = f"D:{os.sep}teste"
        os.makedirs(f"{pat}{os.sep}", exist_ok=True)
        
        for i, (X, y) in enumerate(test_loader):                
            X, y = X.to('cuda').requires_grad_(), y.numpy()
            fig = plt.figure(figsize=(24, 7))
            output = model_ft(X)
            y_pred = (nn.Sigmoid()(output) >= 0.5).int().cpu().numpy().squeeze()

            for j in range(0, list(X.shape)[0]):                                    
                grayscale_cam = cam(input_tensor=X[j].unsqueeze(0), targets=targets)
                grayscale_cam = grayscale_cam[0, :]

                un_normed = un_minus(un_divide(X[j]))
                un_normed = torch.transpose(un_normed, 0, 2).cpu().detach().numpy()

                #un_normed = torch.transpose(X[j], 0, 2).cpu().detach().numpy()
                visualization = show_cam_on_image(un_normed, grayscale_cam, use_rgb=True)            

                fig.add_subplot(rows, columns, j+1, title = 'Y(%i - %s) - Pred(%i - %s)' % (y[j], label_desc[y[j]], 
                                                                            y_pred[j], label_desc[y_pred[j]] ) )
                plt.imshow(visualization)    
            fig.savefig(f"{pat}{os.sep}grad_cam_{name}_{i}.png")