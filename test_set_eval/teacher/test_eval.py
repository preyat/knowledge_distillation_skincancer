
import os
from glob import glob
import torch
from torch import nn
from torch.utils.data import DataLoader
from torchvision import models,transforms
import warnings


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
    save_classification_report
)

#presets
warnings.filterwarnings("ignore", category=UserWarning)
os.environ['CUDA_LAUNCH_BLOCKING'] = '1'


#constants
epoch_num_constant = 5
lesion_type_dict = {
    'nv': 0, #not cacnerous, makes up a lot of the datset 
    'mel':1,
    'bkl': 0,
    'bcc': 1,
    'akiec': 1,
    'vasc': 0,
    'df': 0
}
norm_mean = (224, 224, 3, 20030)
norm_std = [[0.16967013, 0.152372, 0.14070098],[0.570536, 0.5461356, 0.7635292],[0.16967013, 0.152372, 0.14070098]]

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

#paths
data_dir = '../../../test_images'
all_image_path = glob(os.path.join(data_dir, '*', '*.jpg'))
imageid_path_dict = {os.path.splitext(os.path.basename(x))[0]: x for x in all_image_path}


def create_model_mobilenet(feature_extract, use_pretrained=True):
    fine_tuned_model = None
    input_size = 224
    fine_tuned_model = models.mobilenet_v2(pretrained=use_pretrained)
    if feature_extract:
        for parameter in fine_tuned_model.parameters():
            parameter.requires_grad = False
    fine_tuned_model.classifier[1] = nn.Linear(fine_tuned_model.last_channel, 1)
    return fine_tuned_model, input_size
   

def setup_data_loaders( df_val, input_size):
    norm_mean = (0.49139968, 0.48215827, 0.44653124)
    norm_std = (0.24703233, 0.24348505, 0.26158768)

    

    val_transform = transforms.Compose([
        transforms.Resize((input_size, input_size)),
        transforms.ToTensor(),
        transforms.Normalize(norm_mean, norm_std)
    ])

     
    validation_set = SkinCancerDataset(df_val, transform=val_transform)
    val_loader = DataLoader(validation_set, batch_size=32, shuffle=False, num_workers=4)

    return  val_loader

for i in range(0,5):
    
    #load test data
    input_directory = "../../../test_images"
    df_val_fold = original_dataset(input_directory)
    df_val_fold = df_val_fold.reset_index()
    val_loader = setup_data_loaders( df_val_fold, 224)
    results_folder_teacher = "fold_" + str(i) 
    epochs = 5

    #load model
    distilled_model_path = f'model_fold_{i}.pth'  
    distilled_model, _ = create_model_mobilenet(False, True)
    distilled_model.load_state_dict(torch.load(distilled_model_path))
    distilled_model = distilled_model.to(device)
    distilled_model.eval()  # Set the model to evaluation mode

    #evaluate
    if not(os.path.exists(results_folder_teacher)):
        os.makedirs(results_folder_teacher)
    preds, labels = get_predictions_real(val_loader, distilled_model)
    plot_and_save_roc(labels, preds, filename=f"{results_folder_teacher}/roc.png")
    plot_and_save_precision_recall(labels, preds, filename=f"{results_folder_teacher}/precision_recall.png")
    report = save_classification_report(labels, preds, filepath=f"{results_folder_teacher}/classification_report.txt")

    print("Results Saved")

