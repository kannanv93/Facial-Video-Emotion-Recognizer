#import cv2
import numpy as np
import pandas as pd
import os
from tqdm import tqdm, tqdm_notebook
#from joblib import Parallel, delayed
#from skimage.io import imread
#from skimage.transform import resize
#import matplotlib.pyplot as plt

import pandas as pd
import numpy as np
from tqdm import tqdm

from sklearn.model_selection import train_test_split

from sklearn.metrics import accuracy_score

# PyTorch libraries and modules
import torch
from torch.autograd import Variable
from torch.nn import Linear, ReLU, CrossEntropyLoss, Sequential, Conv2d, MaxPool2d, Module, Softmax, BatchNorm2d, Dropout
from torch.optim import Adam, SGD

# torchvision for pre-trained models
from torchvision import models

import shutil, sys   
#from google.colab.patches import cv2_imshow
import copy

import torch.nn as nn

#from __future__ import print_function, division

import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim import lr_scheduler
import numpy as np
import torchvision
from torchvision import datasets, models, transforms
#import matplotlib.pyplot as plt
import time
import os
import copy
#plt.ion()  

from torchvision import transforms
from torchvision import datasets
from torch.utils.data import DataLoader

# Change to fit hardware
BATCH_SIZE = 64
print_every = 50
MODEL = 'VGG16'

data_transforms = {
    'train': transforms.Compose([
        #transforms.RandomResizedCrop(size=256, scale=(0.8, 1.0)),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ]),
    'val': transforms.Compose([
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ]),
}

data_dir = '/scratch/kv942/ccm_proj/datadir_act'
image_datasets = {x: datasets.ImageFolder(os.path.join(data_dir, x),
                                          data_transforms[x])
                  for x in ['train', 'val']}

train_loader = DataLoader(image_datasets['train'],
                          batch_size=BATCH_SIZE,
                          shuffle=True,
                          )
val_loader = DataLoader(image_datasets['val'],
                         batch_size=BATCH_SIZE,
                         shuffle=True,
                         )
# test_loader = DataLoader(image_datasets['test'],
#                          batch_size=BATCH_SIZE,
#                          shuffle=False,
#                          )

dataset_sizes = {x: len(image_datasets[x]) for x in ['train', 'val']}
class_names = image_datasets['train'].classes

def imshow(inp, title=None):
    """Imshow for Tensor."""
    inp = inp.numpy().transpose((1, 2, 0))
    mean = np.array([0.485, 0.456, 0.406])
    std = np.array([0.229, 0.224, 0.225])
    inp = std * inp + mean
    inp = np.clip(inp, 0, 1)
    plt.imshow(inp)
    if title is not None:
        plt.title(title)
    plt.pause(0.001)  # pause a bit so that plots are updated

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def train_model(model, criterion, optimizer, scheduler, num_epochs=25):
    
    # Training steps
    since = time.time()
    best_model_wts = copy.deepcopy(model.state_dict())
    best_acc = 0.0
    
    for epoch in range(num_epochs):
        print("Epoch: ", epoch)
        model.train()
        
        running_loss = 0.0   
        running_corrects = 0
        number_training_steps = 0
        # Iterate over data.
        for batch_id, (inputs, labels) in enumerate(train_loader):
            
            inputs, labels = Variable(inputs).to(device), Variable(labels).to(device)
            outputs = model(inputs)
            model.zero_grad() #same as optimizer.zero_grad()
            
            loss = criterion(outputs, labels)
            running_loss += loss.item() # sum up loss of all minibatches
            
            _, preds = torch.max(outputs, 1)
            running_corrects += torch.sum(preds == labels.data).cpu()
            
            loss.backward()
            optimizer.step()
            number_training_steps+=1
            
            if batch_id % print_every == 0:
                # report performance
                print('Train [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                batch_id * len(inputs), len(train_loader.dataset),
                100. * batch_id / len(train_loader), loss.item()))

        scheduler.step()
        
        epoch_trainloss = running_loss / number_training_steps
        #train_loss_list.append(epoch_trainloss)
        epoch_trainaccuracy = running_corrects.double() / len(train_loader.dataset)
        #train_acc_list.append(epoch_trainaccuracy)
        
        # report performance
        print('Train set: Average loss: {:.4f}, Accuracy: {}/{} ({:.2f}%)'.format(
        epoch_trainloss, running_corrects, len(train_loader.dataset),
        100. * epoch_trainaccuracy))
        
        
    
        # Evaluate after every epoch
        
        model.eval()
        running_loss = 0
        running_corrects = 0

        with torch.no_grad():
            
            number_val_steps = 0
            for batch_id, (inputs, labels) in enumerate(val_loader):
                inputs, labels = Variable(inputs.to(device)), Variable(labels.to(device))
                #print("labels = ", labels)
                outputs = model(inputs)
                #print("outputs = ", outputs)
                loss = criterion(outputs, labels)
                running_loss += loss.item() # sum up loss of all minibatches
                number_val_steps += 1
                
                _, preds = torch.max(outputs, 1)
                running_corrects += torch.sum(preds == labels.data).cpu()
                
            
            epoch_valloss = running_loss / number_val_steps
            #val_loss_list.append(epoch_valloss)
            epoch_valaccuracy = running_corrects.double() / len(val_loader.dataset)
            #val_acc_list.append(epoch_valaccuracy)
            #auc = roc_auc_score(truths, predictions)
            
            print('Validation set: Average loss: {:.4f}, Accuracy: {}/{} ({:.2f}%)\n'.format(
            epoch_valloss, running_corrects, len(val_loader.dataset),
            100. * epoch_valaccuracy))
    
        # deep copy the model
        if epoch_valaccuracy > best_acc:
                best_acc = epoch_valaccuracy
                best_model_wts = copy.deepcopy(model.state_dict())
    
    
    time_elapsed = time.time() - since
    print('Training complete in {:.0f}m {:.0f}s'.format(
        time_elapsed // 60, time_elapsed % 60))
    print('Best val Acc: {:4f}'.format(best_acc))
    
    # load best model weights
    model.load_state_dict(best_model_wts)
    return model


def visualize_model(model, num_images=6):
    was_training = model.training
    model.eval()
    images_so_far = 0
    fig = plt.figure()

    with torch.no_grad():
        for i, (inputs, labels) in enumerate(dataloaders['val']):
            inputs = inputs.to(device)
            labels = labels.to(device)

            outputs = model(inputs)
            _, preds = torch.max(outputs, 1)

            for j in range(inputs.size()[0]):
                images_so_far += 1
                ax = plt.subplot(num_images//2, 2, images_so_far)
                ax.axis('off')
                ax.set_title('predicted: {}'.format(class_names[preds[j]]))
                imshow(inputs.cpu().data[j])

                if images_so_far == num_images:
                    model.train(mode=was_training)
                    return
        model.train(mode=was_training)



model = torchvision.models.vgg16(pretrained=True)

#changing the number of classes
model.classifier[6] = nn.Linear(4096,len(class_names))

print("number of classes = ", len(class_names))

# pretrained_state_dict = torchvision.models.alexnet(pretrained=True).state_dict()
# pretrained_param_names = list(pretrained_state_dict.keys())

# for i, param in enumerate(pretrained_param_names[:-2]): 
#     print("\n", param)

#freezing model layers
# Observe that only parameters of final layer are being optimized as
# opposed to before.

# for name, param in model.features.named_parameters():
#     param.requires_grad = False
#     #print("name =", name, "param =", param)

if torch.cuda.device_count() > 1:
  print("we have multiple GPUs..")
  model = nn.DataParallel(model)
  
model = model.to(device)

num_epochs = 40

criterion = nn.CrossEntropyLoss()

optimizer = optim.SGD(model.parameters(), lr=0.001, momentum=0.9)

# Decay LR by a factor of 0.1 every 7 epochs
exp_lr_scheduler = lr_scheduler.StepLR(optimizer, step_size=7, gamma=0.1)

model = train_model(model, criterion, optimizer, exp_lr_scheduler,num_epochs=num_epochs)

model_folder = '/scratch/sg5783/CCM_Project/final_folder/model_weights/actor_wise_split/'
model_file = model_folder  + MODEL + '_' + str(num_epochs) + '.pth'
torch.save(model.state_dict(), model_file)
