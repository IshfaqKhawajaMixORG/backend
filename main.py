import sys

sys.path.append(".")
path = "."
import warnings
warnings.filterwarnings("ignore")
import torch
from imagebind.models import imagebind_model
from fastapi.staticfiles import StaticFiles
import shutil
import os
from util import *
from get_image_embedding import getImageEmbedding
from get_text_embedding import getTextEmbedding
from fastapi import FastAPI, File, UploadFile
from sklearn.metrics.pairwise import cosine_similarity
from typing import List
import pickle
import numpy as np

model = imagebind_model.imagebind_huge(pretrained=True)
device = "cuda:0" if torch.cuda.is_available() else "cpu"

# Instantiate model
model.eval()
model.to(device)

print("Loaded All Models on device: ", device)


app = FastAPI()


app.mount("/images", StaticFiles(directory="images"), name="images")


# Return Hello From backend on home page
@app.get("/")
def read_root():
    return {"Hello": "From backend"}

# Api to recieve list of images and return list of embeddings
@app.post("/get_image_embeddings")
def get_image_embeddings(files: List[UploadFile] = File(...), paths: List[str] = None):
    try:
        # Delete all files in images folder
        if os.path.exists("./embeddings/image_embeddings.npy"):
            os.remove("./embeddings/image_embeddings.npy")
        # Temporarily save images to images folder
        file_paths = []
        paths = paths[0].split(",")
        for file in files:
            file_path = os.path.join("./images", file.filename)
            with open(file_path, "wb") as file_object:
                shutil.copyfileobj(file.file, file_object)
            file_paths.append(file_path)
        # Get Embeddings
        embeddings = []
        for file,path  in zip(file_paths,paths):
            embeddings.append([path, getImageEmbedding(model, [file], device)[0]])
        # Save embeddings to file using pickle 
        with open("./embeddings/image_embeddings.pkl", "wb") as f:
            pickle.dump(embeddings, f)
        # Remove all files from images folder
        for file in file_paths:
            os.remove(file)
        return {"Success": True}
    except Exception as e:
        print(e)
        return {"Success": False}


  
def cosine_similarity(tensor1, tensor2):
    dot_product = torch.dot(tensor1, tensor2)
    norm_tensor1 = torch.norm(tensor1)
    norm_tensor2 = torch.norm(tensor2)
    similarity = dot_product / (norm_tensor1 * norm_tensor2)
    return similarity.item() 
    


@app.post("/get_text_embeddings")
def get_text_embeddings(text: str,  num_images : int = 5):
    # Get text embedding
    text_embedding = getTextEmbedding(model,[text], device)[0]
    # Read image embeddings from file
    image_embeddings = []
    with open("./embeddings/image_embeddings.pkl", "rb") as f:
        image_embeddings = pickle.load(f)
    # distances = []
    # for img in image_embeddings:
    #     path, embeddings = img[0],img[1]
    #     distances.append([path,torch.dist(text_embedding,embeddings)])
    # distances = sorted(distances, key = lambda x : x[-1])
    # # Return top 5 images
    # images = [distances[i][0] for i in range(num_images)]
    image_tensors = [entry[1] for entry in image_embeddings]
    similarities = [cosine_similarity(text_embedding, img_tensor) for img_tensor in image_tensors]
    top_n_indices = np.argsort(similarities)[::-1][:num_images]
    top_n_image_paths = [image_embeddings[i][0] for i in top_n_indices]
    print(len(similarities))
    print(len(top_n_indices))
    print(num_images)
    print(len(top_n_image_paths))
    print(len(image_embeddings))
    return {"Success": True, "images": top_n_image_paths}



# if __name__ == "__main__":
#     get_text_embeddings(text="cat", num_images=5)