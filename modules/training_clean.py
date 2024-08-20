"""
reference for understanding model training in python :
https://www.kaggle.com/code/ayushraghuwanshi/skin-lesion-classification-acc-90-pytorch-5bb988#Step-3.-Model-training
"""

import numpy as np
import torch
from torch import optim,nn
from torch.autograd import Variable
from torch.utils.data import DataLoader,Dataset
from torchvision import transforms
from PIL import Image
from model import create_model

device = torch.device('cuda:0')

class SkinCancerDataset(Dataset):
    "Dataset class which loads the HAM10000 dataset into a dataframe (df)"
    def __init__(self, df, transform=None):
        self.df = df
        self.transform = transform

    def __len__(self):
        return len(self.df)

    def __getitem__(self, index):
        img_path = self.df['path'][index]
        if img_path is None:
            print(f"No file path available for index {index}, image_id {self.df['image_id'][index]}")
            return None
        image = Image.open(img_path)
        label = torch.tensor(int(self.df['cell_type_idx'][index]))
        if self.transform:
            image = self.transform(image)

        return image,label  
device = torch.device('cuda:0')

class MetricTracker:
    "Keep a track of metrics while training"
    def __init__(self):
        self.reset()

    def reset(self):
        self.value = 0
        self.average = 0
        self.total = 0
        self.count = 0

    def update(self, value, number=1):
        self.value = value
        self.total += value * number
        self.count += number
        self.average = self.total / self.count

train_losses, train_accuracies = [], []
val_losses, val_accuracies = [], []

def process_epoch(model, data_loader, loss_fn, optimizer, epoch_idx, mode='train'):
    "Training or validation epoch code"
    assert mode in ['train', 'validate'], "Mode must be 'train' or 'validate'"

    if mode == 'train':
        model.train()
    else:
        model.eval()

    loss_meter = MetricTracker()
    accuracy_meter = MetricTracker()
    iteration_start = (epoch_idx - 1) * len(data_loader) if mode == 'train' else None

    with torch.set_grad_enabled(mode == 'train'):
        for batch_idx, (imgs, labels) in enumerate(data_loader):
            imgs, labels = Variable(imgs).to(device), Variable(labels).to(device).float()
            if mode == 'train':
                optimizer.zero_grad()
            outputs = model(imgs)
            labels = labels.unsqueeze(1)
            loss = loss_fn(outputs, labels)
            if mode == 'train':
                loss.backward()
                optimizer.step()

            preds = torch.sigmoid(outputs) > 0.5
            correct = preds.eq(labels).sum().item()
            accuracy_meter.update(correct / imgs.size(0))
            loss_meter.update(loss.item())

            if mode == 'train' and (batch_idx + 1) % 100 == 0:
                print(f'Epoch {epoch_idx}, Iteration {batch_idx + 1}/{len(data_loader)}, Loss: {loss_meter.average:.5f}, Accuracy: {accuracy_meter.average:.5f}')
                train_losses.append(loss_meter.average)
                train_accuracies.append(accuracy_meter.average)

    if mode == 'validate':
        print(f'Validation Loss: {loss_meter.average:.5f}, Accuracy: {accuracy_meter.average:.5f}')
        val_losses.append(loss_meter.average)
        val_accuracies.append(accuracy_meter.average)

    return loss_meter.average, accuracy_meter.average



def setup_model_and_data(model_name,  feature_extraction, pretrain=True):
    "Creating model"
    model_out, input_dim = create_model(model_name, feature_extraction, pretrain)
    model = model_out.to(device)
    return model, input_dim

def configure_data_loaders(train_df, val_df, size):
    "Data loaders and preproccesing"
    mean = (0.4914, 0.4822, 0.4465)
    std = (0.2470, 0.2435, 0.2616)
    train_preproccesing = transforms.Compose([
        transforms.Resize((size, size)),
        transforms.RandomHorizontalFlip(),
        transforms.RandomVerticalFlip(),
        transforms.RandomRotation(20),
        transforms.ColorJitter(brightness=0.1, contrast=0.1, saturation=0.1, hue=0.1),
        transforms.ToTensor(),
        transforms.Normalize(mean, std)
    ])
    val_preproccesing = transforms.Compose([
        transforms.Resize((size, size)),
        transforms.ToTensor(),
        transforms.Normalize(mean, std)
    ])

    training_data = SkinCancerDataset(train_df, transform=train_preproccesing)
    validation_data = SkinCancerDataset(val_df, transform=val_preproccesing)
    training_loader = DataLoader(training_data, batch_size=8, shuffle=True, num_workers=4)
    validation_loader = DataLoader(validation_data, batch_size=8, shuffle=False, num_workers=4)

    return training_loader, validation_loader

def train_and_validate_model(model_name, train_df, val_df, num_epochs):
    "training and validation a teacher model"
    model, input_dim = setup_model_and_data(model_name,  feature_extraction=False)
    class_distribution = np.bincount(train_df['cell_type_idx'])
    class_weights = torch.tensor([class_distribution[0] / class_distribution[1]], dtype=torch.float).to(device)

    train_loader, val_loader = configure_data_loaders(train_df, val_df, input_dim)
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    loss_fn = nn.BCEWithLogitsLoss(pos_weight=class_weights).to(device)

    highest_val_accuracy = 0
    for epoch in range(1, num_epochs + 1):
        train_loss, train_accuracy = process_epoch(model, train_loader, loss_fn, optimizer, epoch_idx=1, mode='train')
        val_loss, val_accuracy = process_epoch(model, val_loader, loss_fn, optimizer, epoch_idx=1, mode='validate')

        if val_accuracy > highest_val_accuracy:
            highest_val_accuracy = val_accuracy
            print(f'Best Epoch : Epoch {epoch}, Validation Loss: {val_loss:.5f}, Accuracy: {val_accuracy:.5f}')
    return model,train_loader, val_loader

def train_and_validate_model_student1(model_name, train_df, val_df, num_epochs):
    "training and validation a student model"
    model, input_dim = setup_model_and_data(model_name,  feature_extraction=False)
    class_distribution = np.bincount(train_df['cell_type_idx'])
    class_weights = torch.tensor([class_distribution[0] / class_distribution[1]], dtype=torch.float).to(device)

    train_loader, val_loader = configure_data_loaders(train_df, val_df, 49)
    optimizer = optim.Adam(model.parameters(), lr=0.19)
    loss_fn = nn.BCEWithLogitsLoss(pos_weight=class_weights).to(device)

    highest_val_accuracy = 0
    for epoch in range(1, num_epochs + 1):
        train_loss, train_accuracy = process_epoch(model, train_loader, loss_fn, optimizer, epoch_idx=1, mode='train')
        val_loss, val_accuracy = process_epoch(model, val_loader, loss_fn, optimizer, epoch_idx=1, mode='validate')

        if val_accuracy > highest_val_accuracy:
            highest_val_accuracy = val_accuracy
            print(f'Best Epoch : Epoch {epoch}, Validation Loss: {val_loss:.5f}, Accuracy: {val_accuracy:.5f}')
    return model,train_loader, val_loader

def class_weight(train_df) :
    "Calculating classweights"
    class_distribution = np.bincount(train_df['cell_type_idx'])
    class_weights = torch.tensor([class_distribution[0] / class_distribution[1]], dtype=torch.float).to(device)
    return(class_weights)

def train_and_validate_model_student(model_name, train_df, val_df, num_epochs):
    "training and validating a student model"
    model, input_dim = setup_model_and_data(model_name,  feature_extraction=False)
    class_distribution = np.bincount(train_df['cell_type_idx'])
    class_weights = torch.tensor([class_distribution[0] / class_distribution[1]], dtype=torch.float).to(device)

    train_loader, val_loader = configure_data_loaders(train_df, val_df, input_dim)
    optimizer = optim.Adam(model.parameters(), lr=0.0001)
    loss_fn = nn.BCEWithLogitsLoss(pos_weight=class_weights).to(device)

    highest_val_accuracy = 0
    for epoch in range(1, num_epochs + 1):
        train_loss, train_accuracy = process_epoch(model, train_loader, loss_fn, optimizer, epoch_idx=1, mode='train')
        val_loss, val_accuracy = process_epoch(model, val_loader, loss_fn, optimizer, epoch_idx=1, mode='validate')

        if val_accuracy > highest_val_accuracy:
            highest_val_accuracy = val_accuracy
            print(f'Best Epoch : Epoch {epoch}, Validation Loss: {val_loss:.5f}, Accuracy: {val_accuracy:.5f}')
    return model,train_loader, val_loader


def configure_data_loaders_no_preproccesing(train_df, val_df, size):
    "data loaders with no preproccesing"
    train_preproccessing = transforms.Compose([
        transforms.Resize((size, size)),
        transforms.ToTensor()
    ])
    val_preproccessing = transforms.Compose([
        transforms.Resize((size, size)),
        transforms.ToTensor()
    ])

    training_data = SkinCancerDataset(train_df, transform=train_preproccessing)
    validation_data = SkinCancerDataset(val_df, transform=val_preproccessing)
    training_loader = DataLoader(training_data, batch_size=32, shuffle=True, num_workers=4)
    validation_loader = DataLoader(validation_data, batch_size=32, shuffle=False, num_workers=4)

    return training_loader, validation_loader

def train_and_validate_model_no_preprocessing(model_name, train_df, val_df, num_epochs):
    "Training with no preproccesing"
    model, input_dim = setup_model_and_data(model_name,  feature_extraction=False)
    class_distribution = np.bincount(train_df['cell_type_idx'])
    class_weights = torch.tensor([class_distribution[0] / class_distribution[1]], dtype=torch.float).to(device)

    train_loader, val_loader = configure_data_loaders_no_preproccesing(train_df, val_df, input_dim)
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    loss_fn = nn.BCEWithLogitsLoss(pos_weight=class_weights).to(device)

    highest_val_accuracy = 0
    for epoch in range(1, num_epochs + 1):
        train_loss, train_accuracy = process_epoch(model, train_loader, loss_fn, optimizer, epoch_idx=1, mode='train')
        val_loss, val_accuracy = process_epoch(model, val_loader, loss_fn, optimizer, epoch_idx=1, mode='validate')

        if val_accuracy > highest_val_accuracy:
            highest_val_accuracy = val_accuracy
            print(f'Best Epoch : Epoch {epoch}, Validation Loss: {val_loss:.5f}, Accuracy: {val_accuracy:.5f}')
    return model,train_loader, val_loader
