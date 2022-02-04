import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.utils.data import random_split
from torchvision import datasets, transforms
from network import OdomNet
from dataset import OdomDataset
import os
import numpy as np
from numpy import linalg as LA

torch.manual_seed(1)
torch.autograd.set_detect_anomaly(True)


def save_model(model, filename):
    if not os.path.exists('model'):
        os.mkdir('model')
    print('\n Saving Model ... \n')
    torch.save(model.state_dict(), os.path.join('model', filename))

dataset = OdomDataset('data_pose')

valpercentage = int(np.floor(len(dataset)/10))
trainpercentage = int(len(dataset) - valpercentage)
train, val =random_split(dataset, [trainpercentage, valpercentage])
train_loader = DataLoader(train, batch_size=20,
                    shuffle=True, num_workers=0)
val_loader = DataLoader(val, batch_size=20,
                    shuffle=True, num_workers=0)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

model = OdomNet().to(device)

# criterion = nn.CrossEntropyLoss()
criterion= torch.nn.MSELoss() 
# criterion =  torch.nn.SmoothL1Loss()

optimizer = optim.SGD(model.parameters(), lr=0.005, momentum=0.9)
# optimizer = optim.Adam(model.parameters(), lr=0.003)

best_acc = 0
for epoch in range(30):
    model.train()
    count = 0
    for inputs, label in train_loader:
        inputs = inputs.float().to(device)
        label =  label.float().to(device)
        optimizer.zero_grad()
        output = model(inputs)

        loss = criterion(output,label)
        loss.backward()

        optimizer.step()

        count += len(inputs)
        
        print('Epoch {}: ({}/{})\tLoss: {:.6f}\t\t'.format(epoch, count, len(train_loader.dataset), loss.item()), end='\r')

    model.eval()
    val_loss = 0
    acc = 0
    n_correct = 0
    err_x = 0.
    err_y = 0.

    with torch.no_grad():
        for inputs, label in val_loader:
            inputs = inputs.float().to(device)
            label =  label.float().to(device)
            output = model(inputs)
            val_loss += criterion(output, label).item()

            err_x += abs(output[0][0]-label[0][0])
            err_y += abs(output[0][1]-label[0][1])


            # if (err_x < 0.2 and err_y < 0.2 and err_z < 0.2 and err_roll < 10 and err_pitch < 10 and err_yaw < 10):
            if (abs(output[0][0]-label[0][0]) + abs(output[0][1]-label[0][1]))*(1/2) < 0.1:        
                n_correct += 1
        acc = 100. * n_correct / len(val_loader)
    
    print('\nValidation loss: {:.4f}, Validation accuracy:({:.02f}%)'.format(val_loss, acc))
    print('Erros: x: {:.4f}, y: {:.4f}\n'.format(err_x/len(val_loader), err_y/len(val_loader)))
    if acc > best_acc:
        save_model( model, 'best_model.pth')
        best_acc = acc
    if epoch == 25:    
        save_model( model, 'last_model.pth')