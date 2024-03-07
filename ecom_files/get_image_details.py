import faiss
import numpy as np
import torch
import time
import sys
import os
current_dir = os.path.dirname(__file__)
parent_dir = os.path.abspath(os.path.join(current_dir, os.pardir))
sys.path.append(parent_dir)
from imagebind.models import imagebind_model
from get_image_embedding import getImageEmbedding

def load_faiss_index(index_path):
    return faiss.read_index(index_path)

def load_text_mapping(mapping_path):
    return np.load(mapping_path, allow_pickle=True).item()

def search_texts_from_image_embedding(index, embedding, top_n=5):
    start_time = time.time()    
    embedding = embedding.reshape(1, -1)
    D, I = index.search(embedding, top_n)
    end_time = time.time()
    print(f"Total time to generate FAISS database: {end_time - start_time} seconds")    
    return D, I
def main(image_path, model, device, top_n):
    index_path = 'main.index'
    index = load_faiss_index(index_path)
    image_embedding = getImageEmbedding(model,[image_path], device)
    _, I = search_texts_from_image_embedding(index, image_embedding, top_n=1)
    print("First Level Index Found:  ", I)
    # Get the first cluster
    t = I[0][0]
    # print("First Cluster:  ", t)
    top_n = int(top_n)
    index_path = f'ecom_embeddings/{t}.index'
    # print("index path is : ", index_path)
    index = load_faiss_index(index_path)
    _, I = search_texts_from_image_embedding(index, image_embedding, top_n=top_n)
    # print("Second Level Index Found:  ", I)    
    # Load the text mapping
    tags = []
    for i in I[0]:
        # Load File from text embedding folder:
        try:
            with open(f'ecom_npy_files/{t}-{i}.npy', 'rb') as f:
                # Load the text from the file
                text = np.load(f, allow_pickle=True).item()
                print(text)
                # Check if t contains NaN then fill it with empty string
                for i in text.keys():
                    if text[i] is None:
                        text[i] = ""
                tags.append(t)
        except Exception as e:
            print(e)
    print("\n\n")
    print("-"*100)
    print("Tags Found are : \n ")
    print(tags)
    print("-"*100)
    return tags
if __name__ == "__main__":
    image_path = "image.jpg"  
    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    print("Device   ", device)
    model = imagebind_model.imagebind_huge(pretrained=True)
    model.eval()
    model.to(device)
    main(image_path, model, device)