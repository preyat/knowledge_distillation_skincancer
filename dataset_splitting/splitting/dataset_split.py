import pandas as pd
import os
import shutil
from glob import glob
from sklearn.model_selection import train_test_split

# Define paths
metadata_direction = '../../../input'
source_directory = '../../../input'
destination_directory = '../../../test_images/test_data'
metadata_path = os.path.join(metadata_direction, 'HAM10000_metadata.csv')
os.makedirs(destination_directory, exist_ok=True)

# Read in data
metadata = pd.read_csv(metadata_path)

all_image_paths = glob(os.path.join(source_directory, '*', '*.jpg'))
if not all_image_paths:
    raise ValueError("No images found. Check the path or glob pattern.")

def map_paths_to_ids(df, paths):
    "maps ids from metadat to image paths"
    id_to_path = {os.path.splitext(os.path.basename(path))[0]: path for path in paths}
    df['path'] = df['image_id'].map(id_to_path)
    return df.dropna(subset=['path'])

# Map ids to Paths
mapped_metadata = map_paths_to_ids(metadata, all_image_paths)

if mapped_metadata.empty:
    raise ValueError("No metadata entries matched the image paths.")
mapped_metadata = mapped_metadata.drop_duplicates(subset=['lesion_id'], keep='first')

if len(mapped_metadata) < 2:
    raise ValueError("Not enough data to split. Need more than one entry.")

#Get test set
_, test_df = train_test_split(mapped_metadata, test_size=0.1, stratify=mapped_metadata['dx'])

for _, row in test_df.iterrows():
    src = row['path']
    dst = os.path.join(destination_directory, os.path.basename(src))
    shutil.move(src, dst)
    print(f'Moved {src} to {dst}')

print("Test set creation complete.")
