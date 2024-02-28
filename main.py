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
from get_tags import main as get_tags_from_image_embedding
import concurrent.futures


model = imagebind_model.imagebind_huge(pretrained=True)
device = "cuda" if torch.cuda.is_available() else "cpu"
# device = torch.device("mps")
# Instantiate model
model.eval()
model.to(device)

print("Loaded All Models on device: ", device)


app = FastAPI()


app.mount("/images", StaticFiles(directory="images"), name="images")
app.mount("/embeddings", StaticFiles(directory="embeddings"), name="embeddings")


# Return Hello From backend on home page
@app.get("/")
def read_root():
    return {"Hello": "From backend"}

# Api to recieve list of images and return list of embeddings
@app.post("/get_image_embeddings")
def get_image_embeddings(files: List[UploadFile] = File(...), 
                         paths: List[str] = None, 
                         token : List[str] = None):
    
    try:
        token = token[0]
        # print("Token    ", token)
        embeddingPath = f"./embeddings/{token}.pkl"
        embeddings = []
        alreadyPresent = []
        file_paths = []
        # Delete all files in images folder
        if os.path.exists(embeddingPath):
            # Read it using pickle
            with open(embeddingPath, "rb") as f:
                embeddings = pickle.load(f)
        # print("Already Present length of embeddings", len(embeddings))
        # Temporarily save images to images folder
        sentPaths = paths[0].split(",")
        pathsToSave = []
        if len(embeddings) > 0:
            alreadyPresent = [entry[0] for entry in embeddings]
        for file,path in zip(files,sentPaths):
            if path in alreadyPresent:
                # Remove from paths
                # print("Already present", path)
                continue
            file_path = os.path.join("./images", file.filename)
            with open(file_path, "wb") as file_object:
                shutil.copyfileobj(file.file, file_object)
            file_paths.append(file_path)
            pathsToSave.append(path)
        
        # Get Embeddings
        # Parallelize this
        # allEmbeddings = getImageEmbedding(model, file_paths, device)
        # print("Length of allEmbeddings", len(allEmbeddings))
        # for file,path  in zip(file_paths,pathsToSave):
        #     embeddings.append([path, getImageEmbedding(model, [file], device)[0]])
         # Parallelize the process of getting embeddings
        with concurrent.futures.ThreadPoolExecutor() as executor:
            results = executor.map(lambda x: (x[0], getImageEmbedding(model, [x[1]], device)[0]), zip(pathsToSave, file_paths))
        
        for path, embedding in results:
            embeddings.append([path, embedding])
        print("length of embeddings", len(embeddings))
        # Save embeddings to file using pickle 
        with open(embeddingPath, "wb") as f:
            pickle.dump(embeddings, f)
        # Remove all files from images folder
        for file in file_paths:
            os.remove(file)
        return {
            "Success": True, 
            "total_images": len(embeddings),
            "images": [entry[0] for entry in embeddings]
            }
    except Exception as e:
        print(e)
        return {"Success": False, "Error": str(e)}


  
def cosine_similarity(tensor1, tensor2):
    dot_product = torch.dot(tensor1, tensor2)
    norm_tensor1 = torch.norm(tensor1)
    norm_tensor2 = torch.norm(tensor2)
    similarity = dot_product / (norm_tensor1 * norm_tensor2)
    return similarity.item() 
    


@app.post("/get_text_embeddings")
def get_text_embeddings(text: str,  
                        num_images : int = 5,
                        token : str = None
                        ):
    try:
        # Get text embedding
        text_embedding = getTextEmbedding(model,[text], device)[0]
        # Read image embeddings from file
        image_embeddings = []
        embeddingPath = f"./embeddings/{token}.pkl"
        if not os.path.exists(embeddingPath):
            return {"Success": False, "Error": "No embeddings found"}
        with open(embeddingPath, "rb") as f:
            image_embeddings = pickle.load(f)
        image_tensors = [entry[1] for entry in image_embeddings]
        similarities = [cosine_similarity(text_embedding, img_tensor) for img_tensor in image_tensors]
        temp  = np.argsort(similarities)[::-1]
        top_n_indices = temp[:num_images] if len(temp) > num_images else temp
        top_n_image_paths = [image_embeddings[i][0] for i in top_n_indices]
        print(len(top_n_image_paths))
        return {
            "Success": True,
            "images": top_n_image_paths,
            "total_images": len(image_embeddings),
            }
    except Exception as e:
        print(e)
        return {"Success": False, "Error": str(e)}


@app.post("/get_all_images")
def get_all_images(token : str = None):
    try:
        # Read image embeddings from file
        image_embeddings = []
        embeddingPath = f"./embeddings/{token}.pkl"
        if not os.path.exists(embeddingPath):
            return {"Success": False, "Error": "No embeddings found"}
        with open(embeddingPath, "rb") as f:
            image_embeddings = pickle.load(f)
        image_paths = [entry[0] for entry in image_embeddings]
        return {
            "Success": True,
            "images": image_paths,
            "total_images": len(image_embeddings),
            }
    except Exception as e:
        print(e)
        return {"Success": False, "Error": str(e)}



# Get tags API
@app.post("/image/get_tags")
def get_tags(File : UploadFile = File(...)):
    try:
        # Save the image to images folder
        file_path = os.path.join("./images", File.filename)
        with open(file_path, "wb") as file_object:
            shutil.copyfileobj(File.file, file_object)
        # Get tags
        tags = get_tags_from_image_embedding(file_path, model, device)
        # Remove the file from images folder
        os.remove(file_path)
        return {"Success": True, "tags": tags}
    except Exception as e:
        print(e)
        return {"Success": False, "Error": str(e)}