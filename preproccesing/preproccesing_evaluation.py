
import warnings
warnings.filterwarnings("ignore", category=UserWarning)

import os
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'modules')))
from data_processing_clean import loaders
from training_clean import train_and_validate_model
from evaluation_utils import (
    accuracy,
    get_predictions_real,
    plot_and_save_roc,
    plot_and_save_precision_recall,
    save_classification_report
)

#selecting preproccesing
user_choice = input("Do you want results for the simple or complex preprocessed data? (simple/complex): ")
while user_choice != "simple" and user_choice != "complex" :
    user_choice = input("Invalid input ! Please enter :'simple' or 'complex': ")
if user_choice == "simple" :
    input_directory = "../../input"
    results_folder = "simple_preprocessing_results"
else : 
    input_directory = "../../input_proccessed"
    results_folder = "complex_preprocessing_results"

# training basic resnet model
epochs = 5
training_data, validation_data = loaders(input_directory)
model, train_loader, validation_loader = train_and_validate_model('resnet', training_data, validation_data, epochs)
if not(os.path.exists(results_folder)):
    os.makedirs(results_folder)

#evaluation metrics
preds, labels = get_predictions_real(validation_loader, model)
plot_and_save_roc(labels, preds, filename=f"{results_folder}/roc.png")
plot_and_save_precision_recall(labels, preds, filename=f"{results_folder}/precision_recall.png")
report = save_classification_report(labels, preds, filepath=f"{results_folder}/classification_report.txt")

print("Results Saved")



