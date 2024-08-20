
import warnings
warnings.filterwarnings("ignore", category=UserWarning)

import os
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'modules')))
from data_processing_clean import loaders
from training_clean import train_and_validate_model_no_preprocessing
from evaluation_utils import (
    accuracy,
    get_predictions_real,
    plot_and_save_roc,
    plot_and_save_precision_recall,
    save_classification_report
)


#loading data
input_directory = "../../input"
results_folder = "baseline"
epochs = 5
training_data, validation_data = loaders(input_directory)

#training basic resnet model
model, train_loader, validation_loader = train_and_validate_model_no_preprocessing('resnet', training_data, validation_data, epochs)

#evaluate
if not(os.path.exists(results_folder)):
    os.makedirs(results_folder)
preds, labels = get_predictions_real(validation_loader, model)
plot_and_save_roc(labels, preds, filename=f"{results_folder}/roc.png")
plot_and_save_precision_recall(labels, preds, filename=f"{results_folder}/precision_recall.png")
report = save_classification_report(labels, preds, filepath=f"{results_folder}/classification_report.txt")

print("Results Saved")



