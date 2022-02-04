import os
import torch
from torch import nn
from torch.utils.data import DataLoader
from matplotlib import pyplot as plt
from network import OdomNet, OdomNet2, OdomNet3
from dataset import OdomDataset, OdomDataset2test
import numpy as np

dataset = OdomDataset2test('data_pose',transform=None)
dataloader = DataLoader(dataset, batch_size=1,
                    shuffle=False, num_workers=0)

# fig = plt.figure()

with torch.no_grad():
    

    
    # Loading the saved model
    # save_path = './model/last_model.pth'
    save_path = './model/last_model2.pth'
    # mlp = OdomNet()
    mlp = OdomNet3()
    mlp.load_state_dict(torch.load(save_path))
    mlp.eval()
    
    # for i in range(len(dataset)):
    errx = []
    erry = []   
    rotX_gt = []
    rotY_gt = []
    rotX_p = []
    rotY_p = []
    ant1,ant2,ant3,ant4 = 0,0,0,0   
    for i, data in enumerate(dataloader):
        print("      "+str(i)+"       ",end="\r")
        true_target = data[-1].numpy()

        image = data[0]
        image1 = data[1]
        # Generate prediction
        # prediction = mlp(image)
        prediction = mlp(image,image1)
        
        # Predicted class value using argmax
        predicted_class = prediction.numpy()
        
        # Reshape image
        # image = image.reshape(6, 960, 1280).permute(1,2,0)
        errx.append(abs(predicted_class[0][0]-true_target[0][0]))
        erry.append(abs(predicted_class[0][1]-true_target[0][1]))

        rotX_gt.append(true_target[0][0] + ant1)
        rotY_gt.append(true_target[0][1] + ant2)
        rotX_p.append(predicted_class[0][0] + ant3)
        rotY_p.append(predicted_class[0][1] + ant4)
        ant1 = rotX_gt[i]
        ant2 = rotY_gt[i]
        ant3 = rotX_p[i]
        ant4 = rotY_p[i]

        # plot2x.append()
        # plot2y.append()
        
        # if i == 10:
        #     break
        # Show result
    # plot1 = np.array(plot1)
    # plot2 = np.array(plot2)

    fig, ax = plt.subplots( 1,2)
    # ax[0].imshow(image.numpy())
    # title = f'Prediction: {predicted_class} - Actual target: {true_target}'
    # ax[0].title(title)
    ax[0].plot(errx,label="x")
    ax[0].plot(erry,label="y")
    ax[1].plot(rotX_gt,rotY_gt,label="gt")
    ax[1].plot(rotX_p,rotY_p,label="predicted")
    ax[0].legend()
    ax[1].legend()
    plt.show()