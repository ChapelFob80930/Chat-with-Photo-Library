import os
# model imports
# import json
import torch
# from openai import OpenAI
# import torch.nn as nn
from torch.utils.data import DataLoader
import clip
from dotenv import load_dotenv

load_dotenv()

os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

# helper imports
from tqdm import tqdm
import json
import os
import numpy as np
import pickle
from typing import List, Union, Tuple

# visualisation imports
from PIL import Image
import matplotlib.pyplot as plt
import base64

def initialise_model():
    # Load the CLIP model
    device = "cuda" if torch.cuda.is_available() else "cpu"
        # device = "cpu"
    model, preprocess = clip.load("ViT-B/32", device=device)
    # model, preprocess = clip.load("RN50", device="cpu")
    print(f"Using device: {device}")

    return model, preprocess, device

try:
    
    model, preprocess, device = initialise_model()
    
except Exception as e:
    print("Error loading CLIP model:", e)
    exit(1)