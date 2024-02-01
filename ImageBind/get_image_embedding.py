from imagebind import data
import torch
from imagebind.models import imagebind_model
from imagebind.models.imagebind_model import ModalityType
from util import *

def getImageEmbedding(model,images,device):
  inputs = {
    ModalityType.VISION: load_and_transform_vision_data(images, device),
  }
  with torch.no_grad():
    embeddings = model(inputs)
  return embeddings[ModalityType.VISION]