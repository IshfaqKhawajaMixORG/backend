import faiss
import numpy as np
import torch
import time
from imagebind import data
from imagebind.models import imagebind_model
from imagebind.models.imagebind_model import ModalityType
from get_image_embedding import getImageEmbedding

def load_faiss_index(index_path):
    return faiss.read_index(index_path)

def load_text_mapping(mapping_path):
    return np.load(mapping_path, allow_pickle=True).item()

def generate_embedding_for_input_image(image_path, model, device):
    model.eval()
    inputs = {ModalityType.VISION: data.load_and_transform_vision_data([image_path], device)}
    with torch.no_grad():
        embedding = model(inputs)
        embedding = embedding[ModalityType.VISION].cpu().numpy() if torch.is_tensor(embedding[ModalityType.VISION]) else embedding[ModalityType.VISION]

def search_texts_from_image_embedding(index, embedding, top_n=5):
    start_time = time.time()    
    embedding = embedding.reshape(1, -1)
    D, I = index.search(embedding, top_n)
    # for i, idx in enumerate(I[0]):
    #     print(f"Match {i+1}: {text_mapping[idx]} (Distance: {D[0][i]})")
    end_time = time.time()
    print(f"Total time to generate FAISS database: {end_time - start_time} seconds")    
    return D, I
def main(image_path):
    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    print("Device   ", device)
    model = imagebind_model.imagebind_huge(pretrained=True)
    model.eval()
    model.to(device)

    index_path = 'faiss_data_index.index'
    # mapping_path = 'text_mapping.npy'

    index = load_faiss_index(index_path)
    # text_mapping = load_text_mapping(mapping_path)

    # image_embedding = generate_embedding_for_input_image(image_path, model, device)
    image_embedding = getImageEmbedding(model,[image_path], device)
    print(image_embedding)
    _, I = search_texts_from_image_embedding(index, image_embedding)
    # Load the text mapping
    tags = []
    for i in I[0]:
        # Load File from text embedding folder:
        with open(f'text_embeddings/{i}.npy', 'rb') as f:
            try:
                # Load the text from the file
                text = np.load(f, allow_pickle=True)
                tags.append(text.item())
            except Exception as e:
                print(e)
    
    print("\n".join(tags))
if __name__ == "__main__":
    image_path = "data/n01704323_2389.JPEG"  
    main(image_path)