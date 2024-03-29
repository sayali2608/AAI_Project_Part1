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


classes = ('cloth_mask', 'mask_worn_incorrectly', 'n95_mask', 'no_mask', 'surgical_mask')

imagePath = './dataset/'
transform = transforms.Compose([transforms.Resize((128,128)), transforms.ToTensor()])
data = datasets.ImageFolder(root=imagePath, transform=transform)
loader = DataLoader(dataset=data,batch_size=32)

#Image size normalization

def get_mean_and_std(dataloader):
    channels_sum, channels_squared_sum, num_batches = 0, 0, 0
    for data, _ in dataloader:
        # Mean over batch, height and width, but not over the channels
        channels_sum += torch.mean(data, dim=[0,2,3])
        channels_squared_sum += torch.mean(data**2, dim=[0,2,3])
        num_batches += 1
    
    mean = channels_sum / num_batches

    # std = sqrt(E[X^2] - (E[X])^2)
    std = (channels_squared_sum / num_batches - mean ** 2) ** 0.5

    return mean, std

mean, std = get_mean_and_std(loader)
print('Before normalization:')
print('Mean: '+  str(mean))
print('Standard Dev: '+  str(std))

transform = transforms.Compose([transforms.Resize((128,128)), transforms.ToTensor(),
                                transforms.Normalize(mean=mean,std=std)])
data = datasets.ImageFolder(root=imagePath, transform=transform)
loader = DataLoader(data,batch_size=32)
new_mean, new_std = get_mean_and_std(loader)
print('After normalization:')
print('Mean: '+  str(new_mean))
print('Standard Dev: '+  str(new_std))

train_size = int(0.8 * len(data))
test_size = len(data) - train_size
train_dataset, test_dataset = torch.utils.data.random_split(data, [train_size, test_size], generator=torch.Generator().manual_seed(42))
print('Train size: ' + str(len(train_dataset)))
print('Test size: ' + str(len(test_dataset)))
train_loader = DataLoader(train_dataset,batch_size=32)
test_loader = DataLoader(test_dataset,batch_size=32)

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

cnn = ConvNet()
optimizer = optim.Adam(cnn.parameters(),lr=0.00001)
loss_func = nn.CrossEntropyLoss()
epochs = 1
training_losses = []
validation_losses = []

for e in range(epochs, epochs+2):
    cnn.train()
    training_loss=0
    for i, (batch,labels) in enumerate(train_loader):
        y_h = cnn(batch)
        cnn.zero_grad()
        training_loss = loss_func(y_h,labels)
        training_loss.backward()
        optimizer.step()
    training_losses.append(training_loss)

    print('Epoch:{}, training_loss:{}'.format(e,training_loss,))
epochs+=10

plt.figure()
training_losses = [tl.detach().numpy() if torch.torch.is_tensor(tl) else tl for tl in training_losses]
plt.plot(training_losses,label='Training Loss')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.title('Cross Entropy Loss')
plt.legend()

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

torch.save(model, './MaskNet.pt')

class ConvNet_Variant1(nn.Module):
    def __init__(self):
        super().__init__()
        self.convlayers =  nn.Sequential(
            nn.Conv2d(3,32,3,padding=1), nn.BatchNorm2d(32), nn.LeakyReLU(),
            nn.Conv2d(32,32,3,padding=1), nn.BatchNorm2d(32), nn.LeakyReLU(),
            nn.Conv2d(32,32,3,padding=1), nn.BatchNorm2d(32), nn.LeakyReLU(),
            nn.MaxPool2d(2,2),
            nn.Conv2d(32,64,3,padding=1), nn.BatchNorm2d(64), nn.LeakyReLU(),
            nn.Conv2d(64,64,3,padding=1), nn.BatchNorm2d(64), nn.LeakyReLU(),
            nn.Conv2d(64,64,3,padding=1), nn.BatchNorm2d(64), nn.LeakyReLU(),
            nn.MaxPool2d(2,2),
            nn.Conv2d(64,128,3,padding=1), nn.BatchNorm2d(128), nn.LeakyReLU(),
            nn.Conv2d(128,128,3,padding=1), nn.BatchNorm2d(128), nn.LeakyReLU(),
            nn.Conv2d(128,128,3,padding=1), nn.BatchNorm2d(128), nn.LeakyReLU(),
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

class ConvNet_Variant2(nn.Module):
    def __init__(self):
        super().__init__()
        self.convlayers =  nn.Sequential(
            nn.Conv2d(3,16,3,padding=1),  nn.LeakyReLU(),
            nn.MaxPool2d(2,2),
            nn.Conv2d(16,32,3,padding=1),  nn.LeakyReLU(),
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