import math
import pandas as pd
import matplotlib.pyplot as plt #plotting
import numpy as np
import os #to access directory
from tqdm import tqdm #counting files
import seaborn as sns #visual beautification
import cv2
import io #input/ouput from local
from sklearn.model_selection import train_test_split
from sklearn.utils import shuffle #shuffles matrics randomly with continuation of same pattern
from warnings import filterwarnings #ignore deprecation
from PIL import Image #pillow for image open, rotate and display

import ipywidgets as widgets  #for button
from IPython.display import display, clear_output 

import torch
from torchvision.datasets import ImageFolder
import torchvision.transforms as T
from torch.utils.data import DataLoader
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from collections import Counter
import torch.optim as optim


labels=["Final AD JPEG", "Final CN JPEG", "Final EMCI JPEG", "Final LMCI JPEG", "Final MCI JPEG"]
pathADNI = "C:\\Users\\hassa\\Desktop\\Assignment\\archive\\Alzheimers-ADNI"
#preparing train test Data Together

X_train = []
y_train = []

transform = T.Compose([T.Resize((224,224)), 
                        T.ToTensor(),
                        T.RandomCrop(200),
                        T.RandomHorizontalFlip(True)])
train_dataset = ImageFolder(root = os.path.join(pathADNI, "train"), transform=transform)
test_dataset = ImageFolder(root = os.path.join(pathADNI, "test"), transform=transform)

train_dataset.class_to_idx
idx_to_class = {v: k for k, v in train_dataset.class_to_idx.items()}
print(idx_to_class)

def get_class_distribution(dataset_obj):
    count_dict = {k:0 for k, v in dataset_obj.class_to_idx.items()}

    for _, label_id in dataset_obj:
        label = idx_to_class[label_id]
        count_dict[label] += 1
    return count_dict

def plot_from_dict(dict_obj, plot_title, **kwargs):
    return sns.barplot(data = pd.DataFrame.from_dict([dict_obj]).melt(), x = "variable", y = "value", hue = "variable",**kwargs).set_title(plot_title)


# plt.figure(figsize=(15,8))
# plot_from_dict(get_class_distribution(train_dataset), plot_title="Entire Dataset before doing anything")
trainLoader = DataLoader(train_dataset, batch_size=50, shuffle=True)

testLoader = DataLoader(test_dataset, batch_size=50)

train_images, train_labels = next(iter(trainLoader))

print(train_labels)

class FirstCNNModel(nn.Module):
    def __init__(self,num_classes=5):
        super(FirstCNNModel, self).__init__()

        
        self.features = nn.Sequential(
                    nn.Conv2d(3, 64, kernel_size=11, stride=4, padding=2),
                    nn.ReLU(),
                    nn.MaxPool2d(kernel_size=3, stride=2),
                    nn.Conv2d(64, 192, kernel_size=5, padding=2),
                    nn.ReLU(),
                    nn.MaxPool2d(kernel_size=3, stride=2),
                    nn.Conv2d(192, 384, kernel_size=3, padding=1),
                    nn.ReLU(),
                    nn.Conv2d(384, 256, kernel_size=3, padding=1),
                    nn.ReLU(),
                    nn.Conv2d(256, 256, kernel_size=3, padding=1),
                    nn.ReLU(),
                    nn.MaxPool2d(kernel_size=3, stride=2)
                )
        self.avgpool = nn.AdaptiveAvgPool2d((5, 5))
        self.classifier = nn.Sequential(
                    nn.Dropout(),
                    nn.Linear(256 * 5 * 5, 4096),
                    nn.ReLU(),
                    nn.Dropout(),
                    nn.Linear(4096, 4096),
                    nn.ReLU(),
                    nn.Linear(4096, num_classes)
                )

    def forward(self, x):
        x = self.features(x)
        
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        logits = self.classifier(x)
        #probas = F.softmax(logits, dim=1)

        return logits           

        
    
devicde = torch.device('cuda:0')
model = FirstCNNModel()
model = model.to(device= devicde)

criterion = torch.nn.CrossEntropyLoss(size_average=True)
optimizer = optim.SGD(model.parameters(), lr = 0.01)

num_iterations = math.ceil(len(train_dataset)/50)

test_num_iterations = math.ceil(len(test_dataset)/50)

test_loss = 0.0
training_loss = 0.0
total_correct_test_predictions = 0.0
total_correct_train_predictions = 0.0
total_test_examples = 0.0
for epoch in range(25):
    for i, data in enumerate(trainLoader):

        images, labels = data
        images, labels = images.cuda(), labels.cuda()
        # print(f'Epoch {epoch +1}/2 Step {i +1}/{num_iterations} Images {labels.shape}')
        
        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        # print(epoch, i, loss.item())
        loss.backward()
        optimizer.step()

        training_loss += loss.data.item()
    
    training_loss /= num_iterations
    # print(f'Epoch {epoch +1}/{10} Training Loss: {training_loss}')
        
    num_correct_predictions_per_epoch = 0
    num_examples_per_epoch = 0
    test_accuracy_per_epoch = 0.0
    for i, data in enumerate(testLoader):

        images, labels = data
        images, labels = images.cuda(), labels.cuda()

        outputs = model(images)
        loss = criterion(outputs, labels)

        test_loss += loss.data.item()

        correct = torch.eq(torch.max(F.softmax(outputs), dim =1)[1], labels.view(-1))
        num_correct_predictions_per_epoch += torch.sum(correct).item()
        num_examples_per_epoch += correct.shape[0]
        test_accuracy_per_epoch = 100*(num_correct_predictions_per_epoch/num_examples_per_epoch)

    print(f'The accuracy in Epoch {epoch+1} is: {test_accuracy_per_epoch}')

    test_loss /= test_num_iterations
    print(f'Epoch {epoch +1}/{25} Loss: {test_loss}')
    total_correct_test_predictions +=num_correct_predictions_per_epoch
    print(f'Number of correct predictions per epoch: {num_correct_predictions_per_epoch}')
    







