import numpy as np
import torch
from torch.utils.data import DataLoader
from torchvision import transforms
import torch.nn.functional as F
import warnings
from sklearn.model_selection import StratifiedKFold
from scipy import interp

import os
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../..', 'modules')))
from data_processing_clean import loaders,original_dataset, augment_dataset
from training_clean import train_and_validate_model_student,SkinCancerDataset
from evaluation_utils import (
    accuracy,
    get_predictions_real,
    plot_and_save_roc,
    plot_and_save_precision_recall,
    save_classification_report
)

print("Running")

device = torch.device('cuda:0')
input_directory = "../../../input"

# get data
df_original = original_dataset(input_directory)

warnings.filterwarnings("ignore", category=UserWarning)

def setup_data_loaders(df_train, df_val, input_size):
    "Convert dataframes to dataset reference : https://www.kaggle.com/code/ayushraghuwanshi/skin-lesion-classification-acc-90-pytorch-5bb988#Step-3.-Model-training "
    norm_mean = (0.49139968, 0.48215827, 0.44653124)
    norm_std = (0.24703233, 0.24348505, 0.26158768)

    train_transform = transforms.Compose([
        transforms.Resize((input_size, input_size)),
        transforms.RandomHorizontalFlip(),
        transforms.RandomVerticalFlip(),
        transforms.RandomRotation(20),
        transforms.ColorJitter(brightness=0.1, contrast=0.1, hue=0.1),
        transforms.ToTensor(),
        transforms.Normalize(norm_mean, norm_std)
    ])

    val_transform = transforms.Compose([
        transforms.Resize((input_size, input_size)),
        transforms.ToTensor(),
        transforms.Normalize(norm_mean, norm_std)
    ])

    training_set = SkinCancerDataset(df_train, transform=train_transform)
    train_loader = DataLoader(training_set, batch_size=32, shuffle=True, num_workers=32)
    
    validation_set = SkinCancerDataset(df_val, transform=val_transform)
    val_loader = DataLoader(validation_set, batch_size=32, shuffle=False, num_workers=32)

    return train_loader, val_loader


#5 Fold Cross Validation
num_folds = 5
skf = StratifiedKFold(n_splits=num_folds, shuffle=True)

for fold, (train_idx, val_idx) in enumerate(skf.split(np.zeros(len(df_original)), df_original['cell_type_idx'])):
    print(f"Starting Fold {fold + 1}/{num_folds}")

    #Get data for fold
    df_train_fold = df_original.iloc[train_idx]
    df_val_fold = df_original.iloc[val_idx]
    data_aug_rate = {0: 15, 1: 10}
    df_train_fold = augment_dataset(df_train_fold, data_aug_rate)
    df_train_fold = df_train_fold.reset_index()
    df_val_fold = df_val_fold.reset_index()
    train_loader, val_loader = setup_data_loaders(df_train_fold, df_val_fold, 224)

    results_folder = "fold_" + str(fold) 

    #train model
    epochs = 1
    training_data, validation_data = loaders(input_directory)
    model, train_loader, validation_loader = train_and_validate_model_student('student2', training_data, validation_data, epochs)


    #model evaluation
    if not(os.path.exists(results_folder)):
        os.makedirs(results_folder)

    preds, labels = get_predictions_real(validation_loader, model)
    plot_and_save_roc(labels, preds, filename=f"{results_folder}/roc.png")
    plot_and_save_precision_recall(labels, preds, filename=f"{results_folder}/precision_recall.png")
    report = save_classification_report(labels, preds, filepath=f"{results_folder}/classification_report.txt")

    print("Results Saved")

