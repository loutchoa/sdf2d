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
        
        z = self.fc0(x)
        z = F.relu(z)
        z = self.fc1(z)
        z = F.relu(z)
        z = self.fc2(z)
        z = F.relu(z)
        z = self.fc3(z)
        z = F.relu(z)
        
        return z


def train_model(model, dataloaders, criterion, optimizer, num_epochs):
    
    erreur = 1e-2
    
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
            if not phase:
                model.train()  # Set model to training mode
            else:
                model.eval()   # Set model to evaluate mode

            running_loss = 0.0
            running_corrects = 0

            # Iterate over data.
            for i, dat in enumerate(dataloaders[phase]):
                inputs,objectif = dat  
                inputs = inputs.to(device)
                objectif = objectif.to(device)


                # zero the parameter gradients
                optimizer.zero_grad()
                
                # We normalize inputs
                # First : subtracting min
                max_inputs,_ = inputs[:,1:].max(axis=1)
                min_inputs,_ = inputs[:,1:].min(axis=1)
                inputs[:,1:] = inputs[:,1:] - min_inputs.repeat(12,1).transpose(0,1)
                
                # Then : divide to normalize
                # 0.5 mean magnitude
                '''m = inputs[:,1:].mean(1, keepdim=True)
                inputs = inputs/(2*m)
                '''
                # scaling [0;1]
                inputs = inputs/(max_inputs.repeat(13,1).transpose(0,1) - min_inputs.repeat(13,1).transpose(0,1))                
                
                # forward
                # track history if only in train
                with torch.set_grad_enabled(not phase):

                    predits = model(inputs)
                    #print(predits)
                    
                    # Rescale output
                    # mean 
                    '''predits = predits*2*m + torch.unsqueeze(min_inputs,1)
                    '''
                    # rescale
                    #predits = predits*torch.unsqueeze((max_inputs-min_inputs),1) + torch.unsqueeze(min_inputs,1)
                    
                    # or normalize objectif
                    objectif = (objectif-min_inputs)/(max_inputs - min_inputs)
                    
                    predits = predits.view(objectif.size())
                    
                    loss = criterion(predits, objectif)
                    
                    dist = abs(objectif - predits)

                    # backward + optimize only if in training phase
                    if not phase:
                        loss.backward()
                        optimizer.step()

                # statistics
                #print(loss.item())
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
            if phase and epoch_acc > best_acc:
                best_acc = epoch_acc
                best_model_wts = copy.deepcopy(model.state_dict())
            if phase:
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
net = net.to(device)

# Mean Squared Error (MSE) as our loss function.
critere = nn.MSELoss()

# Choisir pour optimiseur le modèle de descente de gradient stochastique
# L'optimiseur gère l'hyperparamètre de taux d'apprentissage "lr" (learning rate) et celui de memoire
opti = optim.SGD(net.parameters(), lr=0.01, momentum=0.90)


# N is batch size; D_in is input dimension;D_out is output dimension.
N, D_in, D_out = 10000, 13, 1

# Besoin de creer la base de donnees pour tester et mettre à jour le code
# Donnees d'entrees
X = torch.from_numpy(np.load('patch.npy')).float()
#X = torch.randn(N,D_in)


# Donnees objectifs
Y = torch.from_numpy(np.load('objectifs.npy')).float()
#Y = Y.reshape(N,D_out)
#Y = torch.randn(N)

# Mélanger les données
s= np.arange(X.shape[0])
np.random.shuffle(s)

X = X[s]
Y = Y[s]

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

trainloader = torch.utils.data.DataLoader(trainset, batch_size=32, shuffle=False)
testloader = torch.utils.data.DataLoader(testset, batch_size=32, shuffle=False)

dataloaderz = [trainloader, testloader]

# Le nombre d'epochs maximal definit le nombre de passages complets sur la base de donnees d'entraînement
epochs = 100

print('Data Sets charges')

# Entrainement
local_solver, hist = train_model(net, dataloaderz, critere, opti, epochs)

# Load a sample image
example_data, example_result = next(iter(trainloader))
example_data = example_data[0].to(device)
# run the tracing
traced_script_module = torch.jit.trace(local_solver, example_data)
# save the converted model
traced_script_module.save("local_solver.pt")

'''
#Normalize data before putting it in the network  + to.(device)
local_solver.to(device)
input = input.to(device)

# We normalize inputs
max_input = inputs[1:].max()
min_input = inputs[1:].min()
input[1:] = input[1:] - min_input

input = input/(max_input - min_input)

predit = local_solver(inputs)

predit = predit*(max_input-min_input) + min_input
'''
