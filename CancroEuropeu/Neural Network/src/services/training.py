from sklearn.metrics import roc_auc_score, roc_curve, auc
from ..utils.global_info import Info
from tqdm.auto import tqdm
import numpy as np
import torch
import pandas as pd
import math
import matplotlib.pyplot as plt
import os
from torch import nn

class Training:   
    def train_epoch(self, model, trainLoader, optimizer, criterion):
        model.train()
        losses = []
        for X, y in trainLoader:    
            X, y = X.to(Info.Device), y.to(Info.Device)            
            optimizer.zero_grad()
            # (1) Passar os dados pela rede neural (forward)
            output = model(X)                              
            # (2) Calcular o erro da saída da rede com a classe das instâncias (loss)                    
            loss = criterion(output, y)        
            # (3) Usar o erro para calcular quanto cada peso (wi) contribuiu com esse erro (backward)
            loss.backward()
            # (4) Ataulizar os pesos da rede neural
            optimizer.step()        
            losses.append(loss.item())        
        model.eval()
        return np.mean(losses)

    def eval_model(self, model, loader, criterion):
        measures = []
        total = 0
        correct = 0
        losses = []
        with torch.no_grad():
            for X, y in loader:                
                X, y = X.to(Info.Device), y.to(Info.Device)             
                output = model(X)                      
                _, y_pred = torch.max(output, 1)
                total += len(y)
                loss = criterion(output, y)
                losses.append(loss.item())             
                correct += (y_pred == y).sum().cpu().data.numpy()
        measures = {'loss' : np.mean(losses), 'acc' : correct/total}
        return measures

    def train_and_evaluate(self, model, num_epochs, train_loader, dev_loader, optimizer, criterion):                
        max_val_acc = 0
        contAcc = 0
        e_measures = []
        pbar = tqdm(range(1,num_epochs+1))
        for e in pbar:
            train_loss = self.train_epoch(model, train_loader, optimizer, criterion)
            measures_on_train = self.eval_model(model, train_loader, criterion)
            measures_on_dev = self.eval_model(model, dev_loader, criterion)
            measures = {'epoch': e, 'train_loss': train_loss, 'train_acc' : measures_on_train['acc'].round(4), 
                'dev_loss' : measures_on_dev['loss'], 'dev_acc' : measures_on_dev['acc'].round(4) }
            if (max_val_acc < measures_on_dev['acc'].round(4)):
                contAcc = -1
                max_val_acc = measures_on_dev['acc'].round(4)
                torch.save(model.state_dict(), f'{Info.PATH}/state_dict.pt')
            
            Info.Writer.add_scalar(f"{Info.BoardX}/Train/Loss", train_loss, e)
            Info.Writer.add_scalar(f"{Info.BoardX}/Train/Accuracy", measures['train_acc'], e)
            Info.Writer.add_scalar(f"{Info.BoardX}/Validation/Loss", measures['dev_loss'], e)
            Info.Writer.add_scalar(f"{Info.BoardX}/Validation/Accuracy", measures['dev_acc'], e)
            Info.Writer.flush()

            contAcc+= 1
            if contAcc == 20:
                break
            pbar.set_postfix(measures)     
            e_measures += [measures]
        return pd.DataFrame(e_measures), model

    def verify_images(self, test_loader, batch_size, model_ft, label_desc):
        test_iter = iter(test_loader)
        images, labels = next(test_iter)

        fig = plt.figure(figsize=(24, 7))
        rows = 2
        columns = math.ceil(batch_size / rows)
        output = model_ft(images.to(Info.Device))
        _, y_pred = torch.max(output, 1)
        y_pred = y_pred.cpu().data.numpy()
        for i in range(0, columns*rows):    
            img = images[i].permute(1, 2, 0).squeeze()    
            fig.add_subplot(rows, columns, i+1, title = 'Classe %i - %i - %s' % (labels[i], y_pred[i], label_desc[ y_pred[i] ] ) )
            plt.imshow(img)                
        fig.savefig(f"{Info.PATH}/image.png")