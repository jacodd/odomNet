import os
import torch
import numpy as np
from torch.utils.data import Dataset, DataLoader
from torchvision import datasets, transforms
import matplotlib.pyplot as plt
from skimage import io
from PIL import Image
import glob
import csv

class OdomDataset(Dataset):

    def __init__(self):
     
        self.imgs = sorted(glob.glob('Data/imgs/*.jpg'))
        print(self.imgs)
        self.file = 'Data/data.csv'
        self.imu = []
        self.gt = []
        self.readfile()

        self.normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])

    def readfile(self):

        with open(self.file) as csv_file:
            csv_reader = csv.reader(csv_file, delimiter=',')
            line_count = 0
            for row in csv_reader:
                if line_count == 0:
                    # print(f'Column names are {", ".join(row)}')
                    line_count += 1
                else:
                    self.imu.append(row[1:11])
                    self.gt.append(row[11:14])
                    line_count += 1


    def __len__(self):
        return len(self.gt)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()
        # image = self.normalize(io.imread(self.imgs[idx]))
        img = Image.open(self.imgs[idx])
        T = transforms.ToTensor()
        image = T(img)
        imu = np.array(self.imu[idx],dtype=float)
        gt = np.array(self.gt[idx],dtype=float)

        # return sample
        return  image, torch.from_numpy(imu), torch.from_numpy(gt)


if __name__ == "__main__":
    dataset = OdomDataset()
    dataloader = DataLoader(dataset, batch_size=4, shuffle=True, num_workers=0)   

    # for i in range(len(dataset)):
    i = 1
    sample = dataset[i]
    print(sample[1])
    print(sample[2])
    trans = transforms.ToPILImage()
    plt.imshow(trans(sample[0]))
    plt.title(str(i))
    plt.show()
