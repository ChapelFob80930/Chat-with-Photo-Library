# Create and initialise the faiss index
import torch
from torch.utils.data import DataLoader, TensorDataset
from PIL import Image
from app.model import model, preprocess, device
import os
from typing import List
import faiss
from faiss import read_index, write_index

image_datapath = "C:/AI_Assisstant/test_dataset/Images"
caption_datapath = "C:/AI_Assisstant/test_dataset/captions.txt/captions.txt"

def get_image_paths(directory: str, number: int = None) -> List[str]:
    torch.cuda.empty_cache()
    image_paths = []
    count = 0
    for filename in os.listdir(directory):
        if filename.endswith('.jpg'):
            # print(filename)
            image_paths.append(os.path.join(directory, filename))
            if number is not None and count == number:
                return [image_paths[-1]]
            count += 1
    torch.cuda.empty_cache()
    return image_paths


def get_features_smart(image_paths, max_batch=16):
    """
    Extract CLIP image features with dynamic batching to avoid OOM.
    
    image_paths : list of str -> paths to images
    model       : CLIP model
    preprocess  : CLIP preprocess function
    device      : 'cuda' or 'cpu'
    max_batch   : max batch size to try
    """
    torch.cuda.empty_cache()
    # Preprocess all images
    images = [preprocess(Image.open(p).convert("RGB")) for p in image_paths]
    images = torch.stack(images)
    
    dataset = TensorDataset(images)
    
    # Start from max_batch, reduce if OOM occurs
    batch_size = min(max_batch, len(dataset))
    
    while batch_size > 0:
        try:
            dataloader = DataLoader(dataset, batch_size=batch_size)
            all_features = []
            
            with torch.no_grad():
                for (batch,) in dataloader:
                    batch = batch.to(device)
                    features = model.encode_image(batch).float()
                    all_features.append(features.cpu())  # move to CPU to save VRAM
            torch.cuda.empty_cache()        
            return torch.cat(all_features, dim=0)
        
        except RuntimeError as e:
            if "out of memory" in str(e):
                print(f"OOM with batch_size={batch_size}, reducing batch size...")
                batch_size = batch_size // 2
                torch.cuda.empty_cache()
            else:
                raise e
    torch.cuda.empty_cache()
    raise RuntimeError("Could not process images even with batch_size=1")

# Usage
# device = "cuda" if torch.cuda.is_available() else "cpu"
# model, preprocess = clip.load("ViT-B/32", device=device)
if read_index("test_image_index.index") is None:
    direc = "C:/AI_Assisstant/test_dataset/Images"
    image_paths = get_image_paths(direc)

    image_features = get_features_smart(image_paths)
    print(image_features.shape)

    index = faiss.IndexFlatIP(image_features.shape[1])
    # index.add(image_features)
    index.add(image_features/image_features.norm(dim=-1, keepdim=True).cpu().numpy())
    write_index(index, "test_image_index.index")