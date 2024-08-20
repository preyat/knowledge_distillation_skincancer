import torch
from torch.nn.functional import  sigmoid
import numpy as np
from sklearn.metrics import roc_curve, auc, precision_recall_curve, average_precision_score, classification_report
import matplotlib.pyplot as plt

device = torch.device('cuda:0')

def accuracy(val_loader, model, filepath="accuracy.txt"):
    "return accuracy of a model"
    model.eval()
    correctly_identified = 0
    total_images = 0
    with torch.no_grad():
        for _, data in enumerate(val_loader):
            images, labels = data
            images = images.to(device)
            labels = labels.to(device)

            outputs = model(images).squeeze()  
            predicted_labels = (outputs > 0.5).long()  
            total_images += labels.size(0)
            correctly_identified += (predicted_labels == labels).sum().item()

        accuracy = (correctly_identified / total_images) * 100
        accuracy_print = f"Correctly identified = {correctly_identified}, Total images = {total_images}, Accuracy = {accuracy:.2f}%"
        print(accuracy_print)
        with open(filepath, 'w') as file:
            file.write(accuracy_print)
        return accuracy


def get_predictions_real(val_loader, model):
    "get the probabilities a model predicts "
    all_probs = []
    all_labels = []
    model.eval()
    with torch.no_grad():
        for data in val_loader:
            images, labels = data
            images = images.to(device)
            
            outputs = model(images)
            probabilities = sigmoid(outputs).squeeze()
            all_probs.extend(probabilities.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
    return np.array(all_probs), np.array(all_labels)

def plot_and_save_roc(labels, preds, filename="teacher_roc_real.png"):
    "plot roc"
    fpr, tpr, _ = roc_curve(labels, preds)
    roc_auc = auc(fpr, tpr)
    plt.figure()
    plt.plot(fpr, tpr, color='darkorange', lw=2, label='ROC curve (area = %0.2f)' % roc_auc)
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver Operating Characteristic')
    plt.legend(loc="lower right")
    plt.savefig(filename)
    plt.close()

def plot_and_save_precision_recall(labels, preds, filename="teacher_precision_recall_real.png"):
    "plot pr"
    precision, recall, _ = precision_recall_curve(labels, preds)
    #print(precision)
    #print(recall)
    average_precision = average_precision_score(labels, preds)
    area_under_curve = auc(recall, precision)
    print(area_under_curve)
    area_under_line = np.trapz(precision,recall)
    print("Area under the line:", area_under_line)

    # Plotting the Precision-Recall curve
    plt.figure()
    plt.step(recall, precision, where='post', color='b', alpha=0.8, label=f' AUC = {area_under_curve:.2f}')
    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.ylim([0.0, 1.05])
    plt.xlim([0.0, 1.0])
    plt.title('2-class Precision-Recall curve')
    plt.legend(loc="lower left")
    plt.savefig(filename)
    plt.close()

def save_classification_report(labels, preds, filepath='teacher_classification_report_real.txt'):
    "save classification report"
    binary_preds = (preds > 0.5).astype(np.int32)
    report = classification_report(labels, binary_preds)
    with open(filepath, 'w') as file:
        file.write(report)
    return report

def save_classification_report_distillation(labels, preds, filepath='teacher_classification_report_real.txt'):
    "save classification report"
    binary_preds = (preds > 0.7).astype(np.int32)
    report = classification_report(labels, binary_preds)
    with open(filepath, 'w') as file:
        file.write(report)
    return report
