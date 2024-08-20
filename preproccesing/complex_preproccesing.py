import logging
import os
import pandas as pd
from PIL import Image
import cv2
import numpy as np

logging.basicConfig(level=logging.INFO)


def load_and_resize_image(img_path):
    "Load and resize image"
    return Image.open(img_path).convert('RGB').resize((224, 224))

def apply_median_filter(img_np):
    "Apply median filter to decrease noise"
    return cv2.medianBlur(img_np, 5)

def apply_gaussian_filter(img_np):
    "Gaussian blur for noise"
    return cv2.GaussianBlur(img_np, (5, 5), 0)

def remove_hair(img_np):
    "Minimising hair in an image "
    monochrome = cv2.cvtColor(img_np, cv2.COLOR_RGB2GRAY)
    kernel_filt = cv2.getStructuringElement(cv2.MORPH_RECT, (17, 17))
    blackhat = cv2.morphologyEx(monochrome, cv2.MORPH_BLACKHAT, kernel_filt)

    _, blackhat_bin = cv2.threshold(blackhat, 10, 255, cv2.THRESH_BINARY)

    inpainted_img = cv2.inpaint(img_np, blackhat_bin, 1, cv2.INPAINT_TELEA)
    return inpainted_img

def preprocess_and_save(image_path, save_path):
    "Load, proccess and save to knew location"

    img = load_and_resize_image(image_path)
    img_np = np.array(img)
    
    img_np = apply_median_filter(img_np)  
    img_np = apply_gaussian_filter(img_np) 
    img_np = remove_hair(img_np)          

    Image.fromarray(img_np).save(save_path)
    logging.info(f"Processed and saved image: {save_path}")

def process_dataset(metadata_file, image_dir, save_dir):
    "Read metadata, proccess and save image"
    metadata_df = pd.read_csv(metadata_file)

    for _, row in metadata_df.iterrows():
        image_id = row['image_id']
        image_path = os.path.join(image_dir, f"{image_id}.jpg")
        save_path = os.path.join(save_dir, f"{image_id}.jpg")

        if not os.path.exists(image_path):
            logging.warning(f"Image not found: {image_path}")
            continue

        preprocess_and_save(image_path, save_path)

if __name__ == '__main__':
    "Define paths and run code"
    image_dir = "../../input/ham10000_images_part_1"
    metadata_file = "../../input/HAM10000_metadata.csv"
    save_dir = "../../input_proccessed/processed_images"

    process_dataset(metadata_file, image_dir, save_dir)
