
import os, itertools
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from tqdm import tqdm
from glob import glob
from PIL import Image
import torch
from torch import optim,nn
from torch.autograd import Variable
from torch.utils.data import DataLoader,Dataset
from torchvision import models,transforms
import torch.nn.functional as F
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
import warnings
from sklearn.metrics import confusion_matrix, classification_report, roc_curve, auc
import seaborn as sns
from itertools import cycle
from scipy import interp
import torch.nn.functional as F
import torch.nn.functional as F
from sklearn.metrics import roc_curve, auc, precision_recall_curve, average_precision_score
from sklearn.metrics import accuracy_score, classification_report, precision_recall_curve, auc, roc_curve
from sklearn.model_selection import StratifiedKFold


import os
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../../..', 'modules')))
from data_processing_clean import loaders,original_dataset, augment_dataset

from training_clean import train_and_validate_model,SkinCancerDataset
from evaluation_utils import (
    accuracy,
    get_predictions_real,
    plot_and_save_roc,
    plot_and_save_precision_recall,
    save_classification_report,
    save_classification_report_distillation
)

warnings.filterwarnings("ignore", category=UserWarning)

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

#paths
data_dir = '../../../../input'
all_image_path = glob(os.path.join(data_dir, '*', '*.jpg'))
imageid_path_dict = {os.path.splitext(os.path.basename(x))[0]: x for x in all_image_path}



class Net(nn.Module):
    def __init__(self, input_channels,x):
        super(Net, self).__init__()
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
        x = torch.sigmoid(x)  # Sigmoid activation for binary classification
        return x


def evaluate_accuracy_thresholds(model, data_loader, thresholds):
    model.eval()
    all_probs = []
    all_labels = []
    
    with torch.no_grad():
        for images, labels in data_loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            probabilities = torch.sigmoid(outputs).squeeze()
            all_probs.extend(probabilities.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
    
    all_probs = np.array(all_probs)
    all_labels = np.array(all_labels)
    
    for threshold in thresholds:
        preds = (all_probs >= threshold).astype(int)
        accuracy = accuracy_score(all_labels, preds)
        print(f"Threshold: {threshold:.2f}, Accuracy: {accuracy:.4f}")
        print(classification_report(all_labels, preds, target_names=['Class 0', 'Class 1']))

num_folds = 5
skf = StratifiedKFold(n_splits=num_folds, shuffle=True, random_state=42)


df_original =  df_original = original_dataset(data_dir)


for fold, (train_idx, val_idx) in enumerate(skf.split(np.zeros(len(df_original)), df_original['cell_type_idx'])):
    #load datasets
    print(f"Starting Fold {fold + 1}/{num_folds}")
    df_train_fold = df_original.iloc[train_idx]
    df_val_fold = df_original.iloc[val_idx]
    df_train_fold = df_train_fold.reset_index(drop=True)
    df_val_fold = df_val_fold.reset_index(drop=True)

    # load model      
    distilled_model_path = f'distilled_model_fold_distilled_{fold}.pth'  
    distilled_model = Net(3, 1)  
    distilled_model.load_state_dict(torch.load(distilled_model_path))
    distilled_model = distilled_model.to(device)
    distilled_model.eval()  
    norm_mean = (0.485, 0.456, 0.406)
    norm_std = (0.229, 0.224, 0.225)

   
    val_transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(norm_mean, norm_std)
        ])
    validation_set = SkinCancerDataset(df_val_fold, transform=val_transform)
    val_loader = DataLoader(validation_set, batch_size=32, shuffle=False, num_workers=4)

    #evaluate classifcation report at each threshold
    thresholds = np.arange(0, 1.01, 0.05)  
    evaluate_accuracy_thresholds(distilled_model, val_loader, thresholds)
