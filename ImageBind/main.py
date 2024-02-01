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
from typing import List
import numpy as np
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
def get_image_embeddings(files: List[UploadFile] = File(...)):
    # Delete all files in images folder
    shutil.rmtree("./images")
    os.mkdir("./images")
    # Remove image_embeddings.npy if it exists
    if os.path.exists("image_embeddings.npy"):
        os.remove("image_embeddings.npy")
    # Get Embeddings
    embeddings = []
    for file in files:
        # Save file to images folder
        file_path = os.path.join("./images",file.filename)
        with open(file_path, "wb") as file_object:
            shutil.copyfileobj(file.file, file_object)
        # Get Embedding  
        embedding = getImageEmbedding(model,[file_path], device)  
        embeddings.append([file_path,embedding])
    # Save embeddings to file using pickle 
    with open("image_embeddings.pkl", "wb") as f:
        pickle.dump(embeddings, f)
    return {"Success": True}


@app.post("/get_text_embeddings")
def get_text_embeddings(text: str,  num_images : int = 5):
    # Get text embedding
    text_embedding = getTextEmbedding(model,[text], device)
    # Read image embeddings from file
    image_embeddings = []
    with open("image_embeddings.pkl", "rb") as f:
        image_embeddings = pickle.load(f)
    # Calculate  similarity
    distances = []
    for img in image_embeddings:
        path, embeddings = img[0],img[1]
        distances.append([path,torch.dist(text_embedding,embeddings)])
    distances = sorted(distances, key = lambda x : x[-1])
    # Return top 5 images
    images = [distances[i][0][1:] for i in range(num_images)]
    return {"Success": True, "images": images}