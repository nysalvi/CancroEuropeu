from tqdm.auto import tqdm
import numpy as np
import torch
import pandas as pd
import math
import matplotlib.pyplot as plt
import os
from torch import nn

class Training:
    def train_epoch(self, model, trainLoader, optimizer, criterion, device: str):
        model.train()
        losses = []
        for X, y in trainLoader:    
            X, y = X.to(device), y.to(device)            
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

    def eval_model(self, model, loader, device: str):
        measures = []
        total = 0
        correct = 0
        for X, y in loader:                
            X, y = X.to(device), y.to(device)             
            output = model(X)                      
            _, y_pred = torch.max(output, 1)
            total += len(y)
            correct += (y_pred == y).sum().cpu().data.numpy()
        measures = {'acc' : correct/total}
        return measures

    def train_and_evaluate(self, model, num_epochs, train_loader, test_loader, optimizer, criterion, device: str):
        max_val_acc = 0
        e_measures = []
        pbar = tqdm(range(1,num_epochs+1))
        for e in pbar:
            losses = self.train_epoch(model, train_loader, optimizer, criterion, device)
            measures_on_train = self.eval_model(model, train_loader, device)
            measures_on_test = self.eval_model(model, test_loader, device)
            train_loss = np.mean(losses)
            measures = {'epoch': e, 'train_loss': train_loss, 'train_acc' : measures_on_train['acc'].round(4), 'val_acc' : measures_on_test['acc'].round(4) }
            if (max_val_acc < measures_on_test['acc'].round(4)):
                
                max_val_acc = measures_on_test['acc'].round(4)
                torch.save(model.state_dict(), './temp/modelo')

            pbar.set_postfix(measures)     
            e_measures += [measures]

        return pd.DataFrame(e_measures), model

    def verify_images(self, test_loader, batch_size, model_ft, device: str, label_desc, model_name):
        test_iter = iter(test_loader)
        images, labels = next(test_iter)

        fig = plt.figure(figsize=(24, 7))
        rows = 2
        columns = math.ceil(batch_size / rows)

        output = model_ft(images.to(device))
        _, y_pred = torch.max(output, 1)
        y_pred = y_pred.cpu().data.numpy()
        for i in range(0, columns*rows):    
            img = images[i].permute(1, 2, 0).squeeze()    
            fig.add_subplot(rows, columns, i+1, title = 'Classe %i - %i - %s' % (labels[i], y_pred[i], label_desc[ y_pred[i] ] ) )
            plt.imshow(img)
        os.makedirs('output/images', exist_ok=True)  
        fig.savefig(f"./output/images/{model_name}.png")

    def exportModel(self, model_ft, model_name):
        os.makedirs('output/models', exist_ok=True)  
        torch.save(model_ft, f"./output/models/{model_name}.pth")