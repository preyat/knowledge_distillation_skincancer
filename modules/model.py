"""
reference for understanding model initializing modularization :
https://www.kaggle.com/code/ayushraghuwanshi/skin-lesion-classification-acc-90-pytorch-5bb988#Step-3.-Model-training

reference for understanding CNN construction in python : 
https://gitlab.com/muhammad.dawood/lupi_pytorch/-/tree/master/Distillation%20LUPI%20paper%20experiments?ref_type=heads 

"""

import torch
from torch import nn
from torchvision import models
import torch.nn.functional as F

class Student1(nn.Module):
    "Student 1 Model"
    def __init__(self, input_size, output):
        super(Student1, self).__init__()
        self.hidden1 = nn.Linear(7203, 64)  
        self.bn1 = nn.BatchNorm1d(64)
        self.dp1 = nn.Dropout(p=0.3)
        self.hidden2 = nn.Linear(64, 1)  

    def forward(self, x):
        x = x.view(x.size(0), -1) 
        x = F.leaky_relu(self.bn1(self.hidden1(x)))
        x = self.dp1(x)
        x = torch.sigmoid(self.hidden2(x))  
        return x


class Student2(nn.Module):
    "Student 2 Model"
    def __init__(self, input_channels,x):
        super(Student2, self).__init__()
        
  
        self.conv1 = nn.Conv2d(input_channels, 8, kernel_size=3, stride=1, padding=1)
        self.bn1 = nn.BatchNorm2d(8)
        self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2)
        
    
        self.conv2 = nn.Conv2d(8, 16, kernel_size=3, stride=1, padding=1)
        self.bn2 = nn.BatchNorm2d(16)
        self.pool2 = nn.MaxPool2d(kernel_size=2, stride=2)
        
  
        self.conv3 = nn.Conv2d(16, 32, kernel_size=3, stride=1, padding=1)
        self.bn3 = nn.BatchNorm2d(32)
        self.pool3 = nn.MaxPool2d(kernel_size=2, stride=2)
  
        self.flattened_size = 32 * (224 // 8) * (224 // 8)

        self.output = nn.Linear(self.flattened_size, 1)

    def forward(self, x):
        x = self.pool1(F.relu(self.bn1(self.conv1(x))))
        x = self.pool2(F.relu(self.bn2(self.conv2(x))))
        x = self.pool3(F.relu(self.bn3(self.conv3(x))))
        x = x.view(-1, self.flattened_size)
        x = self.output(x)
        x = torch.sigmoid(x)  
        return x

class Student3(nn.Module):
    "Student 3 Model"
    def __init__(self, input_channels,x):
        super(Student3, self).__init__()
       
        self.conv1 = nn.Conv2d(input_channels, 4, kernel_size=3, stride=1, padding=1)
        self.bn1 = nn.BatchNorm2d(4)
        self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2)

        
        self.conv2 = nn.Conv2d(4, 8, kernel_size=3, stride=1, padding=1)
        self.bn2 = nn.BatchNorm2d(8)
        self.pool2 = nn.MaxPool2d(kernel_size=2, stride=2)

      
        self.flattened_size = 8 * (224 // 4) * (224 // 4)

        self.fc1 = nn.Linear(self.flattened_size, 64)  

        self.output = nn.Linear(64, 1)

    def forward(self, x):
        x = self.pool1(F.relu(self.bn1(self.conv1(x))))
        x = self.pool2(F.relu(self.bn2(self.conv2(x))))
        x = x.view(-1, self.flattened_size)
        x = F.relu(self.fc1(x))
        x = self.output(x)
        x = torch.sigmoid(x)  
        return x
    
device = torch.device('cuda:0')
def freeze_model_parameters(model, freeze_parameters):
    "Function to freeze params if needed in transfer learning"
    if freeze_parameters:
        for parameter in model.parameters():
            parameter.requires_grad = False
def create_model(model,  transfer_freeze, pre=True):
    "Initializing a specific model"
    fine_tuned_model = None
    input_size = 224

    if  model == "mobilenet":
        fine_tuned_model = models.mobilenet_v2(pretrained=pre)
        freeze_model_parameters(fine_tuned_model, transfer_freeze)
        fine_tuned_model.classifier[1] = nn.Linear(fine_tuned_model.last_channel, 1)

    elif model == "resnet":
        fine_tuned_model = models.resnet18(pretrained=pre)
        freeze_model_parameters(fine_tuned_model, transfer_freeze)
        fine_tuned_model.fc = nn.Linear(fine_tuned_model.fc.in_features, 1) 


    elif model == "densenet":
        fine_tuned_model = models.densenet121(pretrained=pre)
        freeze_model_parameters(fine_tuned_model, transfer_freeze)
        fine_tuned_model.classifier = nn.Linear(fine_tuned_model.classifier.in_features, 1)

    elif model == "student1":
        print("Initializing Student1 Model")
        fine_tuned_model = Student1(3, 1)  

    elif model == "student2":
        print("Initializing Student2 Model")
        fine_tuned_model = Student2(3, 1)  


    elif model == "student3":
        print("Initializing Student3 Model")
        fine_tuned_model = Student3(3, 1)  

    return fine_tuned_model, input_size
