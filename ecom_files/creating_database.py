import pandas as pd
import numpy as np
import faiss
import time
import torch
import sys
from tqdm import tqdm
import os
import requests
# For Importing libraries from other directories
current_dir = os.path.dirname(__file__)
parent_dir = os.path.abspath(os.path.join(current_dir, os.pardir))
sys.path.append(parent_dir)
from get_image_embedding import getImageEmbedding
from imagebind.models import imagebind_model


def create_database(model, device):
    dataset_name = "amazon data"
    total_directories = os.listdir(f"{parent_dir}/{dataset_name}")
    main_category = []
    sub_category = []
    for directory in total_directories:
        embeddings = []
        try:
            file_path = f"{parent_dir}/{dataset_name}/{directory}"
            df = pd.read_csv(file_path)
            for i in range(len(df)):  
                try:  
                    row = df.iloc[i]
                    image = row['image']
                    # Save Image to the directory
                    image_path = f"images/{i}.jpg"
                    with open(image_path, 'wb') as f:
                        f.write(requests.get(image).content)
                    # Get Embedding
                    embedding = getImageEmbedding(model, [image_path], device).squeeze()
                    # Delete the image
                    os.remove(image_path)
                    # print(embedding)
                    embeddings.append(embedding)
                except Exception as e:
                    print(e)
                    if os.path.exists(image_path):
                        os.remove(image_path)
                    pass
            sub_category.append(torch.mean(torch.stack(embeddings), dim=0))               
        except Exception as e:
            print(e)
            pass
        break




if __name__ == "__main__":
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print("Device   ", device)
    model = imagebind_model.imagebind_huge(pretrained=True)
    model.eval()
    model.to(device)
    create_database(model, device)
    
