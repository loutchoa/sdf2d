# -*- coding: utf-8 -*-
"""
Created on Thu Jul  2 17:31:16 2020

@author: yohan
"""


# Inclure les fonction d'affichage "print"
from __future__ import print_function
# Inclure numpy abregee np
import numpy as np
# Inclure les fonction d'affichage "plot" abregee plt
from matplotlib import pyplot as plt
# Inclure torch
import torch
# Inclure time
import time
# Inclure copy
import copy


#Creation de notre reseau

# Inclure torch.nn abrege nn
import torch.nn as nn
# Inclure torch.nn.functional abrege F
import torch.nn.functional as F
#Inclure torch.optim abrege optim
import torch.optim as optim

# Declaration de la classe pour le reseau de neurones
class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
    
        self.fc0 = nn.Linear(13,128) 
        self.fc1 = nn.Linear(128, 256)
        self.fc2 = nn.Linear(256,128)
        self.fc3 = nn.Linear(128,1)
        
    def forward(self, x):
        x = F.relu(self.fc0(x))
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = F.relu(self.fc3(x))
        "x = x.view(x.size(0),-1)"

        return x


def train_model(model, dataloaders, criterion, optimizer, num_epochs):
    
    erreur = 1e-1
    
    since = time.time()
    all_val = []
    all_train = []
    val_acc_history = []
    J0 = []
    J1 = []
    best_model_wts = copy.deepcopy(model.state_dict())
    best_acc = 0.0

    for epoch in range(num_epochs):
        print('Epoch {}/{}'.format(epoch+1, num_epochs))
        print('-' * 10)

        # Each epoch has a training and validation phase
        for phase in [0, 1]: #train 0 val 1
            if phase == 0:
                model.train()  # Set model to training mode
            else:
                model.eval()   # Set model to evaluate mode

            running_loss = 0.0
            running_corrects = 0

            # Iterate over data.
            for i, dat in enumerate(dataloaders[phase]):
                inputs,objectif = dat
                #Modifier les axes selon la forme de nos donnees
                """inputs = np.swapaxes(inputs, 1, 3)
                inputs = np.swapaxes(inputs, 2, 3)"""                 
                inputs = inputs.to(device)
                objectif = objectif.to(device)

                # zero the parameter gradients
                optimizer.zero_grad()

                # forward
                # track history if only in train
                with torch.set_grad_enabled(phase == 0):

                    predits = model(inputs)
                    loss = criterion(predits, objectif)

                    dist = abs(objectif - predits)

                    # backward + optimize only if in training phase
                    if phase == 0:
                        loss.backward()
                        optimizer.step()

                # statistics
                running_loss += loss.item() * inputs.size(0)
                running_corrects += torch.sum(dist < erreur)

            epoch_loss = running_loss / len(dataloaders[phase].dataset)
            epoch_acc = running_corrects.double() / len(dataloaders[phase].dataset)
            
            if phase==0:
                print('Entrainem. - Loss: {:.4f} Acc: {:.4f}'.format(epoch_loss, epoch_acc*100))
                all_train.append(epoch_acc*100)
                J0.append(epoch_loss)
            else:
                print('Validation - Loss: {:.4f} Acc: {:.4f}'.format(epoch_loss, epoch_acc*100))
                J1.append(epoch_loss)
            # deep copy the model
            if phase == 1 and epoch_acc > best_acc:
                best_acc = epoch_acc
                best_model_wts = copy.deepcopy(model.state_dict())
            if phase == 1:
                val_acc_history.append(epoch_acc*100)
                all_val.append(best_acc*100)
        print()

    time_elapsed = time.time() - since
    print('Training complete in {:.0f}m {:.0f}s'.format(time_elapsed // 60, time_elapsed % 60))
    print('Best val Acc: {:4f}'.format(best_acc*100))
    epochs = list(range(1, num_epochs+1))
    plt.plot(epochs,val_acc_history,label="Val. acc")
    plt.plot(epochs,all_train, label="Train. acc")
    plt.title("Precision du reseau")
    plt.legend()
    plt.xlabel("Nombre d'epochs")
    plt.ylabel("Precision en %")
    plt.show()
    plt.plot(epochs,all_val)
    plt.title("Precision maximale atteinte par le reseau")
    plt.legend()
    plt.xlabel("Nombre d'epochs")
    plt.ylabel("Precision en %")
    plt.show()
    plt.plot(epochs,J0,label= "erreur train")
    plt.plot(epochs,J1, label= "erreur validation")
    plt.title("Erreur de sortie du reseau")
    plt.legend()
    plt.xlabel("Nombre d'epochs")
    plt.ylabel("Valeur de l'erreur")
    plt.show()
    
    # load best model weights
    model.load_state_dict(best_model_wts)
    return model, val_acc_history

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print(device)

net = Net()

# Pour utiliser le GPU !
net.to(device)

# Mean Squared Error (MSE) as our loss function.
criterion = nn.MSELoss()

learning_rate = 1e-3

# Choisir pour optimiseur le modèle de descente de gradient stochastique
# L'optimiseur gère l'hyperparamètre de taux d'apprentissage "lr" (learning rate) et celui de memoire
optimizer = optim.SGD(net.parameters(), lr=learning_rate, momentum=0.9)


# N is batch size; D_in is input dimension;D_out is output dimension.
N, D_in, D_out = 10000, 13, 1

# Besoin de creer la base de donnees pour tester et mettre à jour le code
# Donnees d'entrees
X = torch.randn(N, D_in)

# Donnees objectifs
Y = torch.randn(N, D_out)

# On garde 8000 donnees pour l'apprentissage et 2000 pour les tests
tab = torch.split(X,[8000,2000])
trainset = tab[0]
testset = tab[1]

tab = torch.split(Y,[8000,2000])
trainlab = tab[0]
testlab = tab[1] 

#definition de l'ensemble d'apprentissage
from torch.utils import data
trainset = data.TensorDataset(torch.Tensor(trainset),torch.Tensor(trainlab).type(torch.float))
testset = data.TensorDataset(torch.Tensor(testset),torch.Tensor(testlab).type(torch.float))

trainloader = torch.utils.data.DataLoader(trainset, batch_size=16, shuffle=False)
testloader = torch.utils.data.DataLoader(testset, batch_size=16, shuffle=False)

dataloaderz = [trainloader, testloader]

# Le nombre d'epochs maximal definit le nombre de passages complets sur la base de donnees d'entraînement
epochs = 20

print('Data Sets charges')

# Entrainement
model_ft, hist = train_model(net, dataloaderz, criterion, optimizer, epochs)