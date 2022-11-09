from PIL import Image
import numpy as np
import torch
import torchvision.transforms as transforms

import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

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
loaded_model = torch.load('./saved_model/MaskNet.pt')
mean = [0.5295, 0.4899, 0.4659]
std = [0.2976, 0.2912, 0.2968]

transformer = transforms.Compose([transforms.Resize((128,128)), transforms.ToTensor(), transforms.Normalize(mean=mean,std=std)])
image = Image.open('./test/1.jpg')
#print(np.array(image).shape)
img_tensor = transformer(image).unsqueeze(0)

result = loaded_model(img_tensor)
_, predicted = torch.max(result, 1)
print(result)
print('Predicted: ', classes[predicted])