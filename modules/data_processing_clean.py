"""
reference for understanding HAM10000 layout and loading: 
https://www.kaggle.com/code/ayushraghuwanshi/skin-lesion-classification-acc-90-pytorch-5bb988#Step-3.-Model-training 
"""

import pandas as pd
from glob import glob
import os
from PIL import Image
import torch
from torch.utils.data import Dataset
import torch
from torch.utils.data import Dataset
from sklearn.model_selection import train_test_split

       

def load_metadata(directory):
    "Load metadata into df"
    return pd.read_csv(os.path.join(directory, 'HAM10000_metadata.csv'))

def map_paths_to_ids(dataframe, image_paths):
    "Map image file paths to their corresponding image IDs"
    id_to_path = {os.path.splitext(os.path.basename(path))[0]: path for path in image_paths}
    dataframe['path'] = dataframe['image_id'].map(id_to_path)
    return dataframe.dropna(subset=['path'])


def classify_images(dataframe, classifications):
    "Classify images based on lesion type and map indices"
    dataframe = dataframe.copy()  
    dataframe.loc[:, 'cell_type_idx'] = dataframe['dx'].map(classifications)
    return dataframe

def remove_duplicates(dataframe):
    "Remove duplicates"
    is_duplicate = dataframe['lesion_id'].duplicated(keep=False)
    return dataframe[~is_duplicate]

def split_dataset(dataframe, test_ratio=0.2):
    "Split dataset into training and validation subsets"
    train_df, val_df = train_test_split(dataframe, test_size=test_ratio, random_state=101, stratify=dataframe['cell_type_idx'])
    return train_df.reset_index(drop=True), val_df.reset_index(drop=True)

def augment_dataset(dataframe, augment_rates):
    "Augment the training dataset based on specified rates"
    for label, rate in augment_rates.items():
        subset = dataframe[dataframe['cell_type_idx'] == label]
        repetitions = rate - 1
        dataframe = pd.concat([dataframe] + [subset] * repetitions, ignore_index=True)
    return dataframe


def original_dataset(directory):
    "Given a directory returns the images as a dataframe"
    classifications = {'nv': 0, 'mel': 1, 'bkl': 0, 'bcc': 1, 'akiec': 1, 'vasc': 0, 'df': 0}
    all_image_paths = glob(os.path.join(directory, '*', '*.jpg'))
    metadata = load_metadata(directory)
    mapped_data = map_paths_to_ids(metadata, all_image_paths)
    classified_data = classify_images(mapped_data, classifications)
    unduplicated_data = remove_duplicates(classified_data)
    return(unduplicated_data)

def loaders(directory):
    "Given a directory returns training and validation loaders"
    classifications = {'nv': 0, 'mel': 1, 'bkl': 0, 'bcc': 1, 'akiec': 1, 'vasc': 0, 'df': 0}
    all_image_paths = glob(os.path.join(directory, '*', '*.jpg'))
    metadata = load_metadata(directory)
    mapped_data = map_paths_to_ids(metadata, all_image_paths)
    classified_data = classify_images(mapped_data, classifications)
    unduplicated_data = remove_duplicates(classified_data)
    train_data, validation_data = split_dataset(unduplicated_data)
    augmentation_rates = {0: 15, 1: 10}
    augmented_train_data = augment_dataset(train_data, augmentation_rates)
    return augmented_train_data, validation_data
