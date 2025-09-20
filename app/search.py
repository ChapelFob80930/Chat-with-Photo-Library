#searching logic using either images or code
import re
import os
from model import model, preprocess, device
import base64
from dotenv import load_dotenv
from openai import OpenAI
from config import settings
from indexing import get_features_smart,index
import torch
import clip


load_dotenv()

# os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
client = OpenAI(api_key=settings.openai_api_key)

# image_datapath = "C:/AI_Assisstant/test_dataset/Images"
image_datapath = "C:/AI_Assisstant/app/data/images"
# caption_datapath = "C:/AI_Assisstant/test_dataset/captions.txt/captions.txt"


# pattern =r'[\w]+_[\w]+\.jpg'
# data = []
#image_path = os.path.join(image_datapath, '1000268201_693b08cb0e.jpg')
image_path = os.path.join(image_datapath, 'IMG_20240914_165207.jpg')
# with open('C:/AI_Assisstant/test_dataset/captions.txt/captions.txt', 'r') as file:
#     for line in file:
#         data.append(line)
# def find_entry(data, key, value):
#     for entry in data[1:]:
#         # print("Entry:", entry)  # Debug: print the entire entry
#         # print(re.findall(pattern,entry))
#         # if(re.findall(pattern,entry))!=[]:
#         print(os.path.join(image_datapath,re.findall(pattern,entry)[0]))
#         if os.path.join(image_datapath,re.findall(pattern,entry)[0]) == value:
#             return {"key":value, "description":entry.replace(re.findall(pattern,entry)[0],'').strip()}
#     return None

def encode_image(image_path):
    with open(image_path, 'rb') as image_file:
        encoded_image = base64.b64encode(image_file.read())
        return encoded_image.decode('utf-8')

def image_query(query, image_path):
    response = client.chat.completions.create(
        model='gpt-4.1-mini',
        messages=[
            {
            "role": "user",
            "content": [
                {
                "type": "text",
                "text": query,
                },
                {
                "type": "image_url",
                "image_url": {
                    "url": f"data:image/jpeg;base64,{encode_image(image_path)}",
                },
                }
            ],
            }
        ],
        max_tokens=300,
    )
    # Extract relevant features from the response
    return response.choices[0].message.content
image_query('Write a short label of what is show in this image?', image_path)

def image_search(query, image):
    image_search_embedding = get_features_smart([image_path])
    distances, indices = index.search(image_search_embedding.reshape(1, -1), 2) #2 signifies the number of topmost similar images to bring back
    distances = distances[0]
    indices = indices[0]
    indices_distances = list(zip(indices, distances))
    indices_distances.sort(key=lambda x: x[1], reverse=True)
    return indices_distances

def text_search(query):
    with torch.no_grad():
        torch.cuda.empty_cache()
        text_query_embedding = model.encode_text(clip.tokenize([query]).to(device)).float()
    text_query_embedding_unit_vector = (text_query_embedding/text_query_embedding.norm(dim=-1, keepdim=True)).cpu().numpy()
    # image_search_embedding = get_features_smart([image_path])
    distances, indices = index.search(text_query_embedding_unit_vector, 5) #2 signifies the number of topmost similar images to bring back
    distances = distances[0]
    indices = indices[0]
    indices_distances = list(zip(indices, distances))
    indices_distances.sort(key=lambda x: x[1], reverse=True)
    return indices_distances
        