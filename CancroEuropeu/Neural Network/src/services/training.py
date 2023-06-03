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
    def __init__(self, scheduler, optimizer, num_epochs):
        self.optimizer = optimizer
        self.num_epochs = num_epochs
        self.scheduler = scheduler(optimizer, num_epochs)

    def train_epoch(self, model, trainLoader, criterion):
        model.train()
        losses = []        
        for X, y in trainLoader:    
            X, y = X.to(Info.Device), y.to(Info.Device)            
            # (1) Passar os dados pela rede neural (forward)
            if model._get_name() == 'Inception3':
                output = model(X)[0].squeeze()            
            else:
                output = model(X).squeeze()           
                
            # (2) Calcular o erro da saída da rede com a classe das instâncias (loss)                    
            loss = criterion(output, y.float())        
            # (3) Usar o erro para calcular quanto cada peso (wi) contribuiu com esse erro (backward)
            loss.backward()
            # (4) Ataulizar os pesos da rede neural
            self.optimizer.step()        
            self.optimizer.zero_grad()

            losses.append(loss.item())                
        self.scheduler.step()
        model.eval()
        return np.mean(losses)

    def eval_model(self, model, loader, criterion):        
        total = 0
        correct = 0
        losses = []
        ground_truth = np.zeros(shape=len(loader.dataset))
        prediction = np.zeros(shape=len(loader.dataset))

        cont = 0
        with torch.no_grad():
            for X, y in loader:                
                X, y = X.to(Info.Device), y.to(Info.Device)      
                ground_truth[cont:cont + len(X)] = y.cpu().data.numpy()                
                output = model(X).squeeze()                                      
                y_pred = (nn.Sigmoid()(output) >= 0.5).float()
                prediction[cont:cont + len(X)] = y_pred.cpu().data.numpy()                
                total += len(y)
                loss = criterion(output, y.float())
                losses.append(loss.item())             
                correct += (y_pred == y).sum().cpu().data.numpy()                
                cont+=len(X)
        if Info.SaveType == 'Accuracy':
            metric = correct/total
        elif Info.SaveType == 'FScore':
            #ground_truth = [x.astype(int) for x in ground_truth]
            #prediction = [x.astype(int) for x in prediction]
            metric = fbeta_score(ground_truth, prediction, beta=0.5)        

        measures = {'loss' : np.mean(losses), 'acc' : metric}
        return measures

    def train_and_evaluate(self, model, train_loader, dev_loader, criterion):                
        os.makedirs(Info.PATH, exist_ok=True)
        os.makedirs(Info.BoardX, exist_ok=True)

        max_metric = 0
        contMetric = 0
        e_measures = []
        pbar = tqdm(range(1,self.num_epochs+1))
        for e in pbar:
            train_loss = self.train_epoch(model, train_loader, criterion)
            measures_on_train = self.eval_model(model, train_loader, criterion)
            measures_on_dev = self.eval_model(model, dev_loader, criterion)
            measures = {'epoch': e, 'train_loss': train_loss, 'train_acc' : measures_on_train['acc'].round(4), 
                'dev_loss' : measures_on_dev['loss'], 'dev_acc' : measures_on_dev['acc'].round(4) }
            if (max_metric < measures_on_dev['acc'].round(4)):
                contMetric = -1
                max_metric = measures_on_dev['acc'].round(4)
                torch.save(model.state_dict(), f'{Info.PATH}{os.sep}state_dict.pt')
            
            Info.Writer.add_scalar(f"{Info.Name}{os.sep}Train{os.sep}Loss", train_loss, e)
            Info.Writer.add_scalar(f"{Info.Name}{os.sep}Train{os.sep}{Info.SaveType}", measures['train_acc'], e)
            Info.Writer.add_scalar(f"{Info.Name}{os.sep}Validation{os.sep}Loss", measures['dev_loss'], e)
            Info.Writer.add_scalar(f"{Info.Name}{os.sep}Validation{os.sep}{Info.SaveType}", measures['dev_acc'], e)
            Info.Writer.flush()

            contMetric+= 1
            if contMetric == 20:
                break
            pbar.set_postfix(measures)     
            e_measures += [measures]
        return pd.DataFrame(e_measures), model

    def verify_images(self, test_loader, batch_size, model_ft, label_desc, grad_layer):               
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
        
        os.makedirs(f"{Info.PATH}{os.sep}", exist_ok=True)
        
        for i, (X, y) in enumerate(test_loader):                
            X, y = X.to(Info.Device).requires_grad_(), y.numpy()
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
            fig.savefig(f"{Info.PATH}{os.sep}grad_cam{i}.png")