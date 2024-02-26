import pandas as pd
import numpy as np
import faiss
import time
import torch
from imagebind import data
from imagebind.models import imagebind_model
from imagebind.models.imagebind_model import ModalityType

def generate_embeddings(text, model, device):
    model.eval()
    inputs = {ModalityType.TEXT: data.load_and_transform_text([text], device)}
    with torch.no_grad():
        embedding = model(inputs)
        embedding = embedding[ModalityType.TEXT].cpu().numpy() if torch.is_tensor(embedding[ModalityType.TEXT]) else embedding[ModalityType.TEXT]
    return embedding

def load_data_from_csv(file_path):
    df = pd.read_csv(file_path)
    return df

def create_embeddings_and_save_mappings(df, model, device):
    embeddings = []
    text_mapping = {}  

    for index, row in df.iterrows():  
        text = row['DisplayName']
        # print(text)
        embedding = generate_embeddings(text, model, device)
        embeddings.append(embedding)
        text_mapping[index] = text  

    return embeddings, text_mapping

def create_faiss_index(embeddings):
    start_time = time.time()
    D = embeddings[0].shape[1]
    index = faiss.IndexFlatL2(D)
    embeddings_array = np.vstack(embeddings).astype('float32')
    index.add(embeddings_array)
    end_time = time.time()
    print(f"Total time to generate FAISS database: {end_time - start_time} seconds")
    return index

def main():
    file_path = 'oidv7-class-descriptions.csv'
    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    print("Device   ", device)
    model = imagebind_model.imagebind_huge(pretrained=True)
    model.eval()
    model.to(device)

    df = load_data_from_csv(file_path)
    embeddings, text_mapping = create_embeddings_and_save_mappings(df, model, device)
    
    index = create_faiss_index(embeddings)
    faiss.write_index(index, 'faiss_data_index.index')
    for key,value in text_mapping.items():
        np.save(f'text_embeddings/{key}.npy',value)
        

if __name__ == "__main__":
    main()