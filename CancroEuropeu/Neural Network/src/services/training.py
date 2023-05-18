from sklearn.metrics import roc_auc_score, roc_curve, auc, fbeta_score
from ..utils.global_info import Info
from tqdm.auto import tqdm
import numpy as np
import torch
import pandas as pd
import math
import matplotlib.pyplot as plt
import os
from torch import nn, optim

class Training:   
    def train_epoch(self, model, trainLoader, optimizer, criterion):
        model.train()
        losses = []        
        for X, y in trainLoader:    
            optimizer.zero_grad()
            X, y = X.to(Info.Device), y.to(Info.Device)            
            # (1) Passar os dados pela rede neural (forward)
            output = model(X)          
            # (2) Calcular o erro da saída da rede com a classe das instâncias (loss)                    
            loss = criterion(output, y)        
            # (3) Usar o erro para calcular quanto cada peso (wi) contribuiu com esse erro (backward)
            loss.backward()
            # (4) Ataulizar os pesos da rede neural
            optimizer.step()        
            losses.append(loss.item())        

        scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, 50)
        scheduler.step()
        model.eval()
        return np.mean(losses)

    def eval_model(self, model, loader, criterion):
        measures = []
        total = 0
        correct = 0
        losses = []
        ground_truth = []
        prediction = []
        with torch.no_grad():
            for X, y in loader:                
                X, y = X.to(Info.Device), y.to(Info.Device)      
                ground_truth.append(y)
                output = model(X)                      
                _, y_pred = torch.max(output, 1)
                prediction.append(y_pred)
                total += len(y)
                loss = criterion(output, y)
                losses.append(loss.item())             
                correct += (y_pred == y).sum().cpu().data.numpy()                

        if Info.SaveType == 'Accuracy':
            metric = correct/total
        elif Info.SaveType == 'FScore':
            metric = fbeta_score(ground_truth, prediction, 0.5)

        measures = {'loss' : np.mean(losses), 'acc' : metric}
        return measures

    def train_and_evaluate(self, model, num_epochs, train_loader, dev_loader, optimizer, criterion):                
        max_metric = 0
        contAcc = 0
        e_measures = []
        pbar = tqdm(range(1,num_epochs+1))
        for e in pbar:
            train_loss = self.train_epoch(model, train_loader, optimizer, criterion)
            measures_on_train = self.eval_model(model, train_loader, criterion)
            measures_on_dev = self.eval_model(model, dev_loader, criterion)
            measures = {'epoch': e, 'train_loss': train_loss, 'train_acc' : measures_on_train['acc'].round(4), 
                'dev_loss' : measures_on_dev['loss'], 'dev_acc' : measures_on_dev['acc'].round(4) }
            if (max_metric < measures_on_dev['acc'].round(4)):
                contAcc = -1
                max_metric = measures_on_dev['acc'].round(4)
                torch.save(model.state_dict(), f'{Info.PATH}/state_dict.pt')
            
            Info.Writer.add_scalar(f"{Info.BoardX}/Train/Loss", train_loss, e)
            Info.Writer.add_scalar(f"{Info.BoardX}/Train/{Info.SaveType}", measures['train_acc'], e)
            Info.Writer.add_scalar(f"{Info.BoardX}/Validation/Loss", measures['dev_loss'], e)
            Info.Writer.add_scalar(f"{Info.BoardX}/Validation/{Info.SaveType}", measures['dev_acc'], e)
            Info.Writer.flush()

            contAcc+= 1
            if contAcc == 20:
                break
            pbar.set_postfix(measures)     
            e_measures += [measures]
        return pd.DataFrame(e_measures), model

    def verify_images(self, test_loader, batch_size, model_ft, label_desc):
        from pytorch_grad_cam import GradCAM
        from pytorch_grad_cam.utils.model_targets import ClassifierOutputTarget
        from pytorch_grad_cam.utils.image import show_cam_on_image
        from torchvision import transforms

        cam = GradCAM(model_ft, target_layers=target_layers, use_cuda=True)            
        targets = [ClassifierOutputTarget(1)]
        target_layers = [model_ft.layer4[-1]]

        rows = 2
        columns = math.ceil(batch_size / rows)

        un_divide = transforms.Normalize([0, 0, 0], [1/0.229, 1/0.224, 1/0.225])
        un_minus = transforms.Normalize([-0.485, -0.456, -0.406], [1, 1, 1])
        
        for i, (X, y) in enumerate(test_loader):                
            X, y = X.to(Info.Device).requires_grad_(), y.numpy()
            fig = plt.figure(figsize=(24, 7))
            for j in range(0, columns*rows):    
                output = model_ft(X[j].unsqueeze(0))
                _, y_pred = torch.max(output, 1)
                y_pred = y_pred.cpu().data.numpy()
                grayscale_cam = cam(input_tensor=X[j].unsqueeze(0), targets=targets)
                grayscale_cam = grayscale_cam[0, :]
                un_normed = un_minus(un_divide(X[j]))
                un_normed = torch.transpose(X[j], 0, 2).cpu().detach().numpy()
                visualization = show_cam_on_image(un_normed, grayscale_cam, use_rgb=True)            

                fig.add_subplot(rows, columns, j+1, title = 'Y(%i - %s) - Pred(%i - %s)' % (y[j], label_desc[y[j]], 
                                                                            y_pred[0], label_desc[y_pred[0]] ) )
                plt.imshow(visualization)            

            fig.savefig(f"{Info.PATH}/grad_cam{i}.png")