"""
Reference used for data loading, preproccesing and training understanding:
https://www.kaggle.com/code/ayushraghuwanshi/skin-lesion-classification-acc-90-pytorch-5bb988#Step-3.-Model-training 
"""

import os
import numpy as np
import pandas as pd
from glob import glob
from PIL import Image
import torch
from torch import optim,nn
from torch.autograd import Variable
from torch.utils.data import DataLoader,Dataset
from torchvision import models,transforms
import torch.nn.functional as F
from sklearn.model_selection import train_test_split
import warnings
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import roc_curve, auc, precision_recall_curve, average_precision_score
from sklearn.metrics import accuracy_score, classification_report, precision_recall_curve, auc, roc_curve

import os
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../..', 'modules')))
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

#presets
warnings.filterwarnings("ignore", category=UserWarning)
os.environ['CUDA_LAUNCH_BLOCKING'] = '1'


#constants
epoch_num_constant = 5
lesion_type_dict = {
    'nv': 0,  
    'mel':1,
    'bkl': 0,
    'bcc': 1,
    'akiec': 1,
    'vasc': 0,
    'df': 0
}


device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
data_dir = '../../../input'
all_image_path = glob(os.path.join(data_dir, '*', '*.jpg'))
imageid_path_dict = {os.path.splitext(os.path.basename(x))[0]: x for x in all_image_path}

#undistilled student model
class Student3(nn.Module):
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



# data loading
def setup_data_loaders(df_train, df_val, input_size):
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
    train_loader = DataLoader(training_set, batch_size=32, shuffle=True, num_workers=4)
    
    validation_set = SkinCancerDataset(df_val, transform=val_transform)
    val_loader = DataLoader(validation_set, batch_size=32, shuffle=False, num_workers=4)

    return train_loader, val_loader

#generate fold for cross val
num_folds = 5
skf = StratifiedKFold(n_splits=num_folds, shuffle=True, random_state=42)

#distillation loss function
def distillation_loss(y_pred, y_true, teacher_outputs, t, alpha):
    y_true = y_true.float()
    y_pred = torch.squeeze(y_pred)
    teacher_probs = torch.sigmoid(teacher_outputs)
    teacher_probs = torch.squeeze(teacher_probs)
    classification_loss = F.binary_cross_entropy_with_logits(y_pred, y_true, pos_weight = class_weights )
    student_probs = torch.sigmoid(y_pred)
    teacher_guided_loss = F.binary_cross_entropy(student_probs, teacher_probs, reduction='mean')
    
    combined_loss = (1 - alpha) * classification_loss + alpha * teacher_guided_loss
    return combined_loss

# training and validation models
class MetricTracker(object):
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

total_loss_per_epoch, total_accuracy_per_epoch  = [],[]
def train(loader, model, loss_fn, optimizer, epoch):
    model.train()
    epoch_loss_tracker  = MetricTracker()
    epoch_accuracy_tracker  = MetricTracker()
    curr_iter = (epoch - 1) * len(loader)
    for i, data in enumerate(loader):
        input_images, target_labels  = data
        N = input_images.size(0)
        input_images = Variable(input_images).to(device)
        target_labels  = Variable(target_labels ).to(device).float()

        optimizer.zero_grad()
        outputs = model(input_images)
        target_labels  = target_labels .unsqueeze(1)  


        loss = loss_fn(outputs, target_labels )
        loss.backward()
        optimizer.step()

        predicted_probs = torch.sigmoid(outputs)
        predicted_labels = (predicted_probs > 0.5).float()  
        correct_preds = (predicted_labels == target_labels ).float().sum()
        acc = correct_preds / N
        epoch_accuracy_tracker .update(acc.item())
        epoch_loss_tracker .update(loss.item())
        curr_iter += 1

        if i  % 150 == 0:
            print('[EPOCH :  %d], [ITERATION : %d / %d], [TRAINING LOSS : %.5f], [TRAIN ACC : %.5f]' % (
                epoch, i + 1, len(loader), epoch_loss_tracker .avg, epoch_accuracy_tracker.avg))
            total_loss_per_epoch.append(epoch_loss_tracker .avg)
            total_accuracy_per_epoch .append(epoch_accuracy_tracker.avg)
    return epoch_loss_tracker .avg, epoch_accuracy_tracker .avg

total_loss_validation, total_accuracy_validation = [],[]
def validate(validation_loader, model, loss_function, opt, epoch):
    model.eval()
    validation_loss_tracker  = MetricTracker()
    validation_accuracy_tracker  = MetricTracker()
    with torch.no_grad():
        for i, data in enumerate(validation_loader):
            input_images, target_labels  = data
            N = input_images.size(0)
            input_images = Variable(input_images).to(device)
            target_labels  = Variable(target_labels ).to(device).float()
            outputs = model(input_images)
            target_labels  = target_labels .unsqueeze(1) 
            probabilities = torch.sigmoid(outputs) 
            prediction = probabilities > 0.5 
            validation_accuracy_tracker .update(prediction.eq(target_labels .view_as(prediction)).sum().item()/N)
            validation_loss_tracker.update(loss_function(outputs, target_labels ).item())
    total_loss_validation.append(validation_loss_tracker.avg)
    total_accuracy_validation.append(validation_accuracy_tracker .avg)
    print('[EPOCH : %d], [VALIDATION LOSS : %.5f], [VALIDATION ACCURACY : %.5f]' % (epoch, validation_loss_tracker .avg, validation_accuracy_tracker.avg))
    return validation_loss_tracker.avg, validation_accuracy_tracker.avg

def train_and_validate_model(type_model, training_data, validation_data, epoch_num):
    model, dimensions, device = load_mobilnet_blank( False, True)
    train_loader, val_loader = setup_data_loaders(training_data, validation_data, dimensions)
    opt_function = optim.Adam(model.parameters(), lr=1e-3)
    loss_function = nn.BCEWithLogitsLoss().to(device)
    best_val_acc = 0
    validation_loss, vlaidation_accuracy = [], []

    for epoch in range(1, epoch_num + 1):
        train_l, train_a = train(train_loader, model, loss_function, opt_function, epoch)
        val_l, val_a = validate(val_loader, model, loss_function, opt_function, epoch)
        validation_loss.append(val_l)
        vlaidation_accuracy.append(val_a)
        if val_a > best_val_acc:
            best_val_acc = val_a
            print(f'EPOCH BEST: {epoch},  VALIDATION LOSS {val_l:.5f}],  VALIDATION ACCURACY {val_a:.5f}')
    return model, train_loader, val_loader


def predict(model, image):
    model.eval()
    with torch.no_grad():
        image = Variable(image).to(device)
        output = model(image)
        probability = torch.sigmoid(output)
        return probability

#loading mobile net / teacher model
def load_mobilnet_blank( freeze_parameters, pre=True):
    
    fine_tuned_model = None
    size = 224
    fine_tuned_model = models.mobilenet_v2(pretrained=pre)
    if freeze_parameters:
        for parameter in fine_tuned_model.parameters():
            parameter.requires_grad = False
    fine_tuned_model.classifier[1] = nn.Linear(fine_tuned_model.last_channel, 1)
    
  
    model = fine_tuned_model.to(device)
    return model, size, device

#evlauation functions
def get_predictions_real(val_loader, model):
    all_probs = []
    all_labels = []
    model.eval()
    with torch.no_grad():
        for data in val_loader:
            images, labels = data
            images = Variable(images).to(device)
            
            outputs = model(images)
            probabilities = torch.sigmoid(outputs).squeeze()  # Ensure probabilities are a 1D array
           
            all_probs.extend(probabilities.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

    return np.array(all_probs), np.array(all_labels)

def evaluate_model_hyperparams(model, data_loader,epoch_iter, end_epoch):
    file_path='evaluation_metrics.txt'
    device = torch.device('cuda:0')
    model.eval()
    all_preds = []
    all_labels = []
    with torch.no_grad():
        for images, labels in data_loader:
            images = images.to(device)
            labels = labels.to(device).float().unsqueeze(1)  # Ensure labels are in the correct shape
            outputs = model(images)
            probabilities = torch.sigmoid(outputs)
            predicted_labels = (probabilities > 0.5).float()
            all_preds.extend(predicted_labels.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
    
    
    all_preds = np.array(all_preds)
    all_labels = np.array(all_labels)
    
    accuracy = accuracy_score(all_labels, all_preds)
    report = classification_report(all_labels, all_preds)
    precision, recall, _ = precision_recall_curve(all_labels, all_preds)
    pr_auc = auc(recall, precision)
    fpr, tpr, _ = roc_curve(all_labels, all_preds)
    roc_auc = auc(fpr, tpr)
    
    print("Accuracy: ", accuracy)
    
    with open('evaluation_metrics.txt', 'a') as file:
        file.write(f'Epoch {epoch_iter}: Accuracy: {accuracy}, Area under PR: {pr_auc}, Area under ROC: {roc_auc}\n')
    if end_epoch :     
        with open('evaluation_metrics.txt', 'a') as file:
            file.write(report)
        print("Classification Report:\n", report)
    print(f'Epoch {epoch_iter}: Accuracy: {accuracy}, Area under PR: {pr_auc}, Area under ROC: {roc_auc}')
    return accuracy


def evaluate_model(model, validation_loader,results_folder):
    if not(os.path.exists(results_folder)):
        os.makedirs(results_folder)

    accuracy_score  = accuracy(validation_loader, model, filepath=f"{results_folder}/accuracy.txt")
    preds, labels = get_predictions_real(validation_loader, model)
    plot_and_save_roc(labels, preds, filename=f"{results_folder}/roc.png")
    plot_and_save_precision_recall(labels, preds, filename=f"{results_folder}/precision_recall.png")
    report = save_classification_report_distillation(labels, preds, filepath=f"{results_folder}/classification_report.txt")
    return(report)

# hyperparamter tuning / training distilled mode
def hyperparameter_tuning(train_loader, val_loader, lr_values, temp_values, alpha_values,teacher_model,fold):
    best_accuracy = 0
    trial_number = 0
    best_params = {}
    epoch_acc_8= 0
    epoch_acc_9 = 0


    
    for lr_value in lr_values:
        for temp in temp_values:
            for alpha in alpha_values:
                
                distilled_model = Student3(3, 1)    
                distilled_model.to(device)
                optimizer = optim.RMSprop(distilled_model.parameters(), lr=lr_value) 
                for epoch in range(10):
                    distilled_model.train()
                    for x_batch, y_batch in train_loader:  
                        x_batch, y_batch = x_batch.to(device), y_batch.to(device)
                        with torch.no_grad():
                            teacher_outputs = teacher_model(x_batch)

                        student_outputs = distilled_model(x_batch)
                        loss = distillation_loss(student_outputs, y_batch, teacher_outputs, temp, alpha)
                        optimizer.zero_grad()
                        loss.backward()
                        optimizer.step()
                    report = evaluate_model(distilled_model, val_loader,fold)
                    print(report)
                    #evaluate_model_hyperparams(distilled_model,val_loader, epoch,False)
                if epoch_acc_8 > epoch_acc_9 :
                    accuracy_model = epoch_acc_8
                else : 
                    accuracy_model = epoch_acc_9
                to_save = (f"Fold : {fold}, Accuracy: {accuracy_model}")
                if accuracy_model > best_accuracy:
                    best_accuracy = accuracy_model
                    best_params = {'lr': lr_value, 'temp': temp, 'alpha': alpha}
                evaluate_model(distilled_model, val_loader,fold )
                #evaluate_model_hyperparams(distilled_model,val_loader, epoch,False)
    torch.save(distilled_model.state_dict(), f"distilled_model_{fold}")
    return best_params, best_accuracy



df_original =  df_original = original_dataset(data_dir)

class_counts = np.bincount(df_original['cell_type_idx'])
weight_for_1 = (class_counts[0]*2) / class_counts[1]
class_weights = torch.tensor([weight_for_1], dtype=torch.float).to(device)

for fold, (train_idx, val_idx) in enumerate(skf.split(np.zeros(len(df_original)), df_original['cell_type_idx'])):
    "Training the teacher and disitlled model across 5 folds"
    
    df_train_fold = df_original.iloc[train_idx]
    df_val_fold = df_original.iloc[val_idx]

    data_aug_rate = {0: 15, 1: 10}
    df_train_fold = augment_dataset(df_train_fold, data_aug_rate)
    df_train_fold = df_train_fold.reset_index()
    df_val_fold = df_val_fold.reset_index()

    train_loader, val_loader = setup_data_loaders(df_train_fold, df_val_fold, 224)

    results_folder_teacher = "fold_" + str(fold) 
    epochs = 5

    teacher_model, train_loader, validation_loader = train_and_validate_model('mobilenet', df_train_fold, df_val_fold, epochs)


    if not(os.path.exists(results_folder_teacher)):
        os.makedirs(results_folder_teacher)

    accuracy_score  = accuracy(validation_loader, teacher_model, filepath=f"{results_folder_teacher}/accuracy.txt")
    preds, labels = get_predictions_real(validation_loader, teacher_model)
    plot_and_save_roc(labels, preds, filename=f"{results_folder_teacher}/roc.png")
    plot_and_save_precision_recall(labels, preds, filename=f"{results_folder_teacher}/precision_recall.png")
    report = save_classification_report(labels, preds, filepath=f"{results_folder_teacher}/classification_report.txt")

    torch.save(teacher_model.state_dict(), f"teacher_model_{fold}")
    print("Results Saved")
    lr_values = [0.0001]
    temp_values = [5]
    alpha_values = [0.25]
    #alpha_values = [0.25]
    

    best_params, best_accuracy = hyperparameter_tuning(train_loader, val_loader, lr_values, temp_values, alpha_values,teacher_model,("fold_distilled_"+str(fold)))
    print("Best Validation Accuracy:", best_accuracy)

