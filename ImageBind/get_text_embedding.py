from imagebind import data
import torch
from imagebind.models import imagebind_model
from imagebind.models.imagebind_model import ModalityType
from util import *
def getTextEmbedding(model, text, device):
  inputs = {
    ModalityType.TEXT: load_and_transform_text(text, device),
  }
  with torch.no_grad():
    embeddings = model(inputs)
  return embeddings[ModalityType.TEXT]