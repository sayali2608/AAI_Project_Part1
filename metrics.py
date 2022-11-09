import os
import numpy as np
import torch
from torch.utils.data import DataLoader
import torchvision.transforms as transforms
import torchvision.datasets as datasets
from PIL import Image
import xml.etree.ElementTree as ET

#For model
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

#For shwing images and graphs
from matplotlib import pyplot as plt

from sklearn.metrics import confusion_matrix, multilabel_confusion_matrix
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

class ConvNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.convlayers =  nn.Sequential(
            nn.Conv2d(3,32,3,padding=1), nn.BatchNorm2d(32), nn.LeakyReLU(),
            nn.MaxPool2d(2,2),
            nn.Conv2d(32,64,3,padding=1), nn.BatchNorm2d(64), nn.LeakyReLU(),
            nn.MaxPool2d(2,2),
            nn.Conv2d(64,128,3,padding=1), nn.BatchNorm2d(128), nn.LeakyReLU(),
            nn.MaxPool2d(2,2)
        )
        self.FC = nn.Sequential(
            nn.Linear(128*16*16,100),
            nn.Linear(100,5)
            
        )
    
    def forward(self,X):
        X = self.convlayers(X)
        X =  self.FC(X.reshape(-1,128*16*16))
        return X

classes = ('cloth_mask', 'mask_worn_incorrectly', 'n95_mask', 'no_mask', 'surgical_mask')
cnn = torch.load('./saved_model/MaskNet.pt')
mean = [0.5295, 0.4899, 0.4659]
std = [0.2976, 0.2912, 0.2968]

imagePath = './test_samples/'

transform = transforms.Compose([transforms.Resize((128,128)), transforms.ToTensor(),
                                transforms.Normalize(mean=mean,std=std)])
data = datasets.ImageFolder(root=imagePath, transform=transform)
test_loader = DataLoader(data,batch_size=32)


org_labels = []
pred_labels  = []
cnn.eval()
with torch.no_grad():
    for  i,(X,y) in enumerate(test_loader):
        pred_y = cnn(X).cpu()
        pred_y = torch.argmax(F.softmax(pred_y),dim=1)
        #print(pred_y)
        if org_labels == []:
            org_labels=y[:]
            pred_labels = pred_y[:]
            
        else:
            org_labels = torch.hstack([org_labels, y])
            pred_labels = torch.hstack([pred_labels, pred_y])

print('Accuracy: ',(org_labels==pred_labels).sum()/len(org_labels))
print('Precision: ', precision_score(org_labels,pred_labels,average='weighted'))
print('Recall: ', recall_score(org_labels,pred_labels,average='weighted'))
print('F1: ', f1_score(org_labels,pred_labels,average='weighted'))
print(confusion_matrix(org_labels,pred_labels))