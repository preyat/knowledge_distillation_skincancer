#Reference for understanding voting : https://www.kaggle.com/code/ayushraghuwanshi/skin-lesion-classification-acc-90-pytorch-5bb988#Step-3.-Model-training
import warnings
warnings.filterwarnings("ignore", category=UserWarning)


import os
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import warnings
from sklearn.metrics import classification_report, roc_curve, auc, precision_recall_curve, average_precision_score


import os
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../..', 'modules')))
from data_processing_clean import loaders
from training_clean import train_and_validate_model
from evaluation_utils import (
    accuracy,
    get_predictions_real,
    plot_and_save_roc,
    plot_and_save_precision_recall,
    save_classification_report
)


def single_model_eval (validation_loader, model, results_folder="results"):
    preds, labels = get_predictions_real(validation_loader, model)
    plot_and_save_roc(labels, preds, filename=f"{results_folder}/roc.png")
    plot_and_save_precision_recall(labels, preds, filename=f"{results_folder}/precision_recall.png")
    report = save_classification_report(labels, preds, filepath=f"{results_folder}/classification_report.txt")

def double_model_eval(validation_loader, model_1, model_2, filepath="results"):
    #setup models
    model_1.eval()
    model_2.eval()

    with torch.no_grad():
        #setup variables
        correctly_identified = 0
        all_labels = []
        all_probabilities_1 = []
        all_probabilities_2 = []

        #iterate over validation set
        for _, data in enumerate(validation_loader):
            images, labels = data
            images = images.to(device)
            labels = labels.to(device)

            #get outputs
            outputs_1 = model_1(images).squeeze() 
            outputs_2 = model_2(images).squeeze()

            #get probabilities 
            probabilities_1 = torch.sigmoid(outputs_1)
            probabilities_2 = torch.sigmoid(outputs_2)

            #get labels
            predicted_labels_1 = (outputs_1 > 0.5).long()  
            predicted_labels_2 = (outputs_2 > 0.5).long()

            #store labels
            all_labels.extend(labels.cpu().numpy())

            #store probabilities
            all_probabilities_1.extend(probabilities_1.cpu().numpy())
            all_probabilities_2.extend(probabilities_2.cpu().numpy())

            #iterate over labels
            for i in range(labels.size(0)):
                #compare outputs
                if predicted_labels_1[i] == predicted_labels_2[i]:
                    correctly_identified += int(labels[i] == predicted_labels_1[i])
                else:
                    confidence_1 = max(outputs_1[i], 1 - outputs_1[i])
                    confidence_2 = max(outputs_2[i], 1 - outputs_2[i])
                    
                    if confidence_1 > confidence_2:
                        correctly_identified += int(labels[i] == predicted_labels_1[i])
                    else:
                        correctly_identified += int(labels[i] == predicted_labels_2[i])

    #Evaluate and store results
    results_path = os.path.join(filepath)
    if not os.path.exists(results_path):
        os.makedirs(results_path)

    # Classification report
    report = classification_report(all_labels, (np.array(all_probabilities_1) + np.array(all_probabilities_2)) / 2 > 0.5, target_names=['Class 0', 'Class 1'], output_dict=True)
    df_report = pd.DataFrame(report).transpose()
    df_report.to_csv(os.path.join(results_path, "classification_report.csv"))

    # ROC curve 
    fpr, tpr, _ = roc_curve(all_labels, (np.array(all_probabilities_1) + np.array(all_probabilities_2)) / 2)
    roc_auc = auc(fpr, tpr)
    plt.figure()
    plt.plot(fpr, tpr, color='darkorange', label='ROC curve (area = %0.2f)' % roc_auc)
    plt.plot([0, 1], [0, 1], color='navy', linestyle='--')
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver Operating Characteristic - Combined Model')
    plt.legend(loc="lower right")
    plt.savefig(os.path.join(results_path, "roc_curve.png"))

    # Precision-Recall curve
    precision, recall, _ = precision_recall_curve(all_labels, (np.array(all_probabilities_1) + np.array(all_probabilities_2)) / 2)
    pr_auc = auc(recall, precision)
    plt.figure()
    plt.step(recall, precision, where='post', color='b', alpha=0.8, label='Area Under PR Curve = %0.2f' % pr_auc)
    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.ylim([0.0, 1.05])
    plt.xlim([0.0, 1.0])
    plt.title('Precision-Recall curve - Combined Model')
    plt.legend(loc="best")
    plt.savefig(os.path.join(results_path, "precision_recall_curve.png"))


def combined_three_models_acc(val_loader, model_1, model_2, model_3, results_folder):
    #setup models
    model_1.eval()
    model_2.eval()
    model_3.eval()

    #setup vars
    correctly_identified_1 = 0
    correctly_identified_2 = 0
    correctly_identified_3 = 0
    correctly_identified_combined = 0
    total_images = 0
    all_labels = []
    all_probabilities_1 = []
    all_probabilities_2 = []
    all_probabilities_3 = []
    
    with torch.no_grad():
        #iterate over labels
        for _, data in enumerate(val_loader):
            images, labels = data
            images = images.to(device)
            labels = labels.to(device)

            #get and store outputs
            outputs_1 = model_1(images).squeeze()
            outputs_2 = model_2(images).squeeze()
            outputs_3 = model_3(images).squeeze()

            #get  probabilities 
            probabilities_1 = torch.sigmoid(outputs_1)
            probabilities_2 = torch.sigmoid(outputs_2)
            probabilities_3 = torch.sigmoid(outputs_3)

            #get predictions
            predicted_labels_1 = (probabilities_1 > 0.5).long()
            predicted_labels_2 = (probabilities_2 > 0.5).long()
            predicted_labels_3 = (probabilities_3 > 0.5).long()

            #store data
            all_labels.extend(labels.cpu().numpy())
            all_probabilities_1.extend(probabilities_1.cpu().numpy())
            all_probabilities_2.extend(probabilities_2.cpu().numpy())
            all_probabilities_3.extend(probabilities_3.cpu().numpy())


            for i in range(labels.size(0)):
                #combining and voting
                correctly_identified_1 += int(labels[i] == predicted_labels_1[i])
                correctly_identified_2 += int(labels[i] == predicted_labels_2[i])
                correctly_identified_3 += int(labels[i] == predicted_labels_3[i])

                combined_decision = (predicted_labels_1[i] + predicted_labels_2[i] + predicted_labels_3[i]) > 1
                correctly_identified_combined += int(labels[i] == combined_decision)
                total_images += 1


    results_path = os.path.join(results_folder)
    if not os.path.exists(results_path):
        os.makedirs(results_path)


    # Classification report
    combined_probabilities = (np.array(all_probabilities_1) + np.array(all_probabilities_2) + np.array(all_probabilities_3)) / 3
    report = classification_report(all_labels, combined_probabilities > 0.5, target_names=['Class 0', 'Class 1'], output_dict=True)
    df_report = pd.DataFrame(report).transpose()
    df_report.to_csv(os.path.join(results_path, "classification_report.csv"))

    # roc curve
    fpr, tpr, _ = roc_curve(all_labels, combined_probabilities)
    roc_auc = auc(fpr, tpr)
    plt.figure()
    plt.plot(fpr, tpr, color='darkorange', label='ROC curve (area = %0.2f)' % roc_auc)
    plt.plot([0, 1], [0, 1], color='navy', linestyle='--')
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver Operating Characteristic - Combined Three Models')
    plt.legend(loc="lower right")
    plt.savefig(os.path.join(results_path, "roc_curve.png"))

    #pr curve
    precision, recall, _ = precision_recall_curve(all_labels, combined_probabilities)
    pr_auc = auc(recall, precision)
    plt.figure()
    plt.step(recall, precision, where='post', color='b', alpha=0.8, label='Area Under PR Curve = %0.2f' % pr_auc)
    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.ylim([0.0, 1.05])
    plt.xlim([0.0, 1.0])
    plt.title('Precision-Recall curve - Combined Three Models')
    plt.legend(loc="best")
    plt.savefig(os.path.join(results_path, "precision_recall_curve.png"))


device = torch.device('cuda:0')
input_directory = "../../../input"
results_folder = "temp"
epochs = 5

# model training
training_data, validation_data = loaders(input_directory)
mobilenet_model, train_loader, validation_loader = train_and_validate_model('mobilenet', training_data, validation_data, epochs)
resnet_model, train_loader, validation_loader = train_and_validate_model('resnet', training_data, validation_data, epochs)
densenet_model, train_loader, validation_loader = train_and_validate_model('densenet', training_data, validation_data, epochs)


#model evaluation
if not(os.path.exists(results_folder)):
    os.makedirs(results_folder)

print ("Accuracy ResNet")
single_model_eval(validation_loader, resnet_model, "res")
print ("Accuracy DenseNet")
single_model_eval(validation_loader, densenet_model, "dense")
print ("Accuracy MobileNet")
single_model_eval(validation_loader, mobilenet_model, "mobile")
print ("Accuracy Res-DenseNet")
double_model_eval(validation_loader, resnet_model, densenet_model, "res_dense")
print ("Accuracy Res-MobileNet")
double_model_eval(validation_loader, resnet_model, mobilenet_model, "res_mobile")
print ("Accuracy Dense-MobileNet")
double_model_eval(validation_loader, densenet_model, mobilenet_model, "dense_mobile")
print ("Accuracy Res-Mobile-DenseNet")
combined_three_models_acc(validation_loader, resnet_model, mobilenet_model, densenet_model, "res_dense_mobile")

if not(os.path.exists("models")):
    os.makedirs("models")
# Save the entire models
torch.save(resnet_model, os.path.join("models", 'resnet_model.pth'))
torch.save(densenet_model, os.path.join("models", 'densenet_model.pth'))
torch.save(mobilenet_model, os.path.join("models", 'mobilenet_model.pth'))
