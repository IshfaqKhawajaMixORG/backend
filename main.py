import sys

sys.path.append(".")
path = "."
import warnings
warnings.filterwarnings("ignore")
import torch
from imagebind.models import imagebind_model
from fastapi.staticfiles import StaticFiles
import json
import shutil
import os
from util import *
from get_image_embedding import getImageEmbedding
from get_text_embedding import getTextEmbedding
from fastapi import FastAPI, File, UploadFile
from sklearn.metrics.pairwise import cosine_similarity
from typing import List
import pickle

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
            embeddings.append([path, getImageEmbedding(model, [file], device)])
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

# def cosineSimilarity(text_embedding, image_embeddings, num_images = 5):
#     similarities = cosine_similarity(text_embedding, image_embeddings)
#     print(similarities)
#     indices = similarities.argsort()[0][::-1][:num_images]
#     print(indices)
#     images = [image_embeddings[i][0] for i in indices]
#     return images
    
    


@app.post("/get_text_embeddings")
def get_text_embeddings(text: str,  num_images : int = 5):
    # Get text embedding
    text_embedding = getTextEmbedding(model,[text], device)
    # Read image embeddings from file
    image_embeddings = []
    with open("./embeddings/image_embeddings.pkl", "rb") as f:
        image_embeddings = pickle.load(f)
    # Calculate  similarity
    # images = cosineSimilarity(text_embedding, image_embeddings, num_images)
    distances = []
    for img in image_embeddings:
        path, embeddings = img[0],img[1]
        distances.append([path,torch.dist(text_embedding,embeddings)])
    distances = sorted(distances, key = lambda x : x[-1])
    # Return top 5 images
    images = [distances[i][0] for i in range(num_images)]
    return {"Success": True, "images": images}