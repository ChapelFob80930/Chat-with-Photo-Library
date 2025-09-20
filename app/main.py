import json
import shutil
from importlib.metadata import metadata
from typing import List, TypedDict

import faiss
import gradio as gr
from PIL import Image
import numpy as np
import re
import os
import time

from matplotlib import interactive
# from gradio.themes.builder_app import chatbot
# from setuptools.sandbox import save_path
from torch.cuda import seed_all

# from app.chatbot import selected_gpu
from model import model, preprocess, device
import base64
from dotenv import load_dotenv
from openai import OpenAI
from config import settings
from indexing import get_features_smart,index
import torch
import clip
from torch.utils.data import DataLoader, TensorDataset
from faiss import read_index, write_index
# from dotenv import load_dotenv
# load_dotenv()



# ----- Placeholder: Your real CLIP functions here ----- #
def embed_text(text):
    with torch.no_grad():
        torch.cuda.empty_cache()
        text_query_embedding = model.encode_text(clip.tokenize([text]).to(device)).float()
    text_query_embedding_unit_vector = (
                text_query_embedding / text_query_embedding.norm(dim=-1, keepdim=True)).cpu().numpy()
    return text_query_embedding_unit_vector

def embed_image(image, max_batch=16):
    torch.cuda.empty_cache()
    # Preprocess all images
    # images = [preprocess(Image.open(p).convert("RGB")) for p in image_paths]
    print(type(image))
    print(image)
    images = [preprocess(image.convert("RGB"))]
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
    raise RuntimeError("Could not process images even with batch_size=1") # TODO: Replace with real image embedding


class Message(TypedDict):
    role: str
    content: str

# ----- Placeholder: Your real prebuilt-index (loaded from disk) ----- #
# PREBUILT_IMAGES = [("pre_img1.jpg", Image.new("RGB", (128, 128))), ("pre_img2.jpg", Image.new("RGB", (128,128)))]
# PREBUILT_EMBEDDINGS = [np.random.randn(512), np.random.randn(512)]
PREBUILT_INDEX = read_index("./index/app_image_index.index")

class PhotoSession:
    def __init__(self):
        self.index_size = None
        self.images = []
        self.embeddings = []
        self.index = None
        self.metadata = []
        self.client = OpenAI(api_key=settings.OPENAI_API_KEY)
        self.chat_history: List[Message] = []
        self.metadata_selected_gpu = None
        self.index_name = None

    def set_prebuilt(self):
        self.index = read_index(f"./index/{self.index_name if self.index_name else 'app_image_index.index'}")
        self.index_size = self.index.ntotal

    def clear(self):
        self.images.clear()
        self.embeddings.clear()
        # self.metadata.clear()
        # self.metadata_selected_gpu.clear()
        # self.chat_history.clear()

session = PhotoSession()



direc = "C:/AI_Assisstant/app/data/images" #TODO: Make environment variable
direc_custom = "./data/uploaded_images" #TODO: Make environment variable

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

def get_image_paths_custom(index: List[int] = None) -> List[str]:
    torch.cuda.empty_cache()
    image_paths = []
    for idx in index:
        metadata = session.metadata[idx]
        image_paths.append(metadata["path"])
    # image_paths = []
    # count = 0
    # for filename in os.listdir(directory):
    #     if filename.endswith('.jpg'):
    #         # print(filename)
    #         image_paths.append(os.path.join(directory, filename))
    #         if number is not None and count == number:
    #             return [image_paths[-1]]
    #         count += 1
    torch.cuda.empty_cache()
    return image_paths

#TODO: Implement function to return images when embedding order in db and image order in file is different
# def get_image_paths(directory: str, number: int = None) -> List[str]:
#
#     torch.cuda.empty_cache()
#     image_paths = []
#     count = 0
#
#     for filename in os.listdir(directory):
#         if number is None:
#             break
#
#         if filename.endswith('.jpg'):
#             # print(filename)
#             if count < number:
#                 count += 1
#
#             else:
#                 break
#
#             image_paths.append(os.path.join(directory, filename))
#
#     torch.cuda.empty_cache()
#     return image_paths

def set_mode(use_prebuilt_index):
    if use_prebuilt_index:
        session.set_prebuilt()
        return gr.update(interactive=False, visible=False), "Using prebuilt demo index with {} images.".format(session.index_size), gr.update(interactive=True, visible=True), gr.update(visible=False), gr.update(interactive=False, visible=False), gr.update(interactive=False, visible=False), gr.update(visible=False), gr.update(visible=True), gr.update(value = None), gr.update(value = None), gr.update(value = None)
    else:
        session.clear()
        return gr.update(interactive=True, visible=True), "Custom Index exists enter your query below" if os.path.exists("./index/custom_image_index.index") and os.path.exists(
                "./index/custom_metadata.json") and len(os.listdir(direc_custom)) > 0 else "Upload your own images to index (in-memory, not saved).", gr.update(interactive=False, visible=False), gr.update(visible=False), gr.update(interactive=False, visible=False), gr.update(interactive=False, visible=False), gr.update(visible=True), gr.update(visible=False), gr.update(value = None), gr.update(value = None), gr.update(value = None)

def upload_images(files):
    session.clear()
    count = 0
    if files is None or len(files) == 0:
        return "No files uploaded."
    for f in files:
        save_path = f"./data/uploaded_images/uploaded_image_{count}.jpg"
        img = Image.open(f)
        img.save(save_path)
        session.metadata.append({
            "filename": f.name,
            "path": save_path,
            "index": count
        })
        session.images.append((f.name, img))
        session.embeddings.append(embed_image(img))
        count+=1
        # print(session.embeddings)
    session.index = faiss.IndexFlatIP(session.embeddings[0].shape[1])
    for embedding in session.embeddings:
        session.index.add(embedding/embedding.norm(dim=-1, keepdim=True).cpu().numpy())
        session.index_size = session.index.ntotal
        write_index(session.index, "C:/AI_Assisstant/app/index/custom_image_index.index")
    with open("./index/custom_metadata.json", "w") as f:
        json.dump(session.metadata, f)
    return "Uploaded and indexed {} images, ready for search.".format(len(session.images))



def create_app_metadata():
    if not os.path.exists(direc):
        print(f"Directory {direc} does not exist.")
        return
    image_paths = get_image_paths(direc)
    metadata = []
    for idx, path in enumerate(image_paths):
        metadata.append({
            "filename": os.path.basename(path),
            "path": path,
            "index": idx
        })
    with open("./index/app_metadata.json", "w") as f:
        json.dump(metadata, f)
    session.metadata = metadata

def search_images(query_mode, text_query, image_query, k, index_name):
    match_indexes = None
    index_load = None
    if os.path.exists(f"./index/{index_name}"):
        index_load = read_index(f"./index/{index_name}")
    if index_name == "app_image_index.index":
        if os.path.exists("./index/app_metadata.json"):
            with open("./index/app_metadata.json", "r") as f:
                session.metadata = json.load(f)
        else:
            create_app_metadata()

    elif index_name == "hardware_index.index":
        if os.path.exists("./index/hardware_metadata.json"):
            with open("./index/hardware_metadata.json", "r") as f:
                session.metadata = json.load(f)
        else:
            if not os.path.exists("./data/hardware_images"):
                print("Hardware images directory does not exist.")
                return [], "No indexed images available. Upload or select demo mode."
            image_paths = get_image_paths("./data/hardware_images")
            metadata = []
            for idx, path in enumerate(image_paths):
                metadata.append({
                    "filename": os.path.basename(path),
                    "path": path,
                    "index": idx
                })
            with open("./index/hardware_metadata.json", "w") as f:
                json.dump(metadata, f)
            session.metadata = metadata

    if not index_load:
        return [], "No indexed images available. Upload or select demo mode."
    if query_mode == "Text":
        query_emb = embed_text(text_query)
        distances, indices = index_load.search(query_emb, k)  # 2 signifies the number of topmost similar images to bring back
        distances = distances[0]
        indices = indices[0]
        indices_distances = list(zip(indices, distances))
        indices_distances.sort(key=lambda x: x[1], reverse=True)
        match_indexes = indices_distances
        # return indices_distances

    else:
        img = Image.open(image_query)
        print(img)
        query_emb = embed_image(img)
        distances, indices = index_load.search(query_emb.reshape(1, -1),k)  # 2 signifies the number of topmost similar images to bring back
        distances = distances[0]
        indices = indices[0]
        indices_distances = list(zip(indices, distances))
        indices_distances.sort(key=lambda x: x[1], reverse=True)
        match_indexes = indices_distances
        # return indices_distances

    # indexes = [idx for idx, distance in match_indexes]
    # image_paths = get_image_paths_custom(indexes)

    images = []
    for (idx, dist) in match_indexes:
        # print(idx)
        if idx >= len(session.metadata) or idx<0:
            break
        meta = session.metadata[idx]
        img = Image.open(meta["path"])
        if index_name == "hardware_index.index":
            # Extract description from filename
            caption = re.sub(r'[_\-]', ' ', os.path.splitext(meta["filename"])[0])
            # caption = f"{description}\nknown_issues: {meta.get('known_issues', 'N/A')[0]}\nsolutions: {meta.get('solutions', 'N/A')[0]}"
        else:
            caption = f"score={dist:.2f}"
        images.append((img, caption))


    return images, f"Returned {len(match_indexes)} result(s).", gr.update(visible=True) if index_name=="hardware_index.index" else gr.update(visible=False), gr.update(interactive = True, visible=True) if index_name=="hardware_index.index" else gr.update(interactive = False, visible=False), gr.update(interactive = True, visible=True) if index_name=="hardware_index.index" else gr.update(interactive = False, visible=False)


def search_images_custom(query_mode, text_query, image_query, k):
    match_indexes = None
    index_load = None
    if os.path.exists("./index/custom_image_index.index"):
        index_load = read_index("./index/custom_image_index.index")
    if os.path.exists("./index/custom_metadata.json"):
        with open("./index/custom_metadata.json", "r") as f:
            session.metadata = json.load(f)
    if not index_load or len(os.listdir(direc_custom))==0:
        return [], "No indexed images available. Upload or select demo mode."


    if query_mode == "Text":
        query_emb = embed_text(text_query)
        distances, indices = index_load.search(query_emb, k)  # 2 signifies the number of topmost similar images to bring back
        distances = distances[0]
        indices = indices[0]
        indices_distances = list(zip(indices, distances))
        indices_distances.sort(key=lambda x: x[1], reverse=True)
        match_indexes = indices_distances
        # return indices_distances

    else:
        img = Image.open(image_query)
        query_emb = embed_image(img)
        distances, indices = index_load.search(query_emb.reshape(1, -1),k)  # 2 signifies the number of topmost similar images to bring back
        distances = distances[0]
        indices = indices[0]
        indices_distances = list(zip(indices, distances))
        indices_distances.sort(key=lambda x: x[1], reverse=True)
        match_indexes = indices_distances
        # return indices_distances

    # indexes = [idx for idx, distance in match_indexes]
    # image_paths = get_image_paths_custom(indexes)

    images = []
    for (idx, dist) in match_indexes:
        # print(idx)
        if idx >= len(session.metadata) or idx<0:
            break
        meta = session.metadata[idx]
        img = Image.open(meta["path"])
        caption = f"score={dist:.2f}"
        images.append((img, caption))


    return images, f"Returned {len(match_indexes)} result(s).", gr.update(visible=False), gr.update(interactive = False, visible=False), gr.update(interactive = False, visible=False)

def run_search(query_mode, text_query, image_query, use_prebuilt, k, index_name):
    if use_prebuilt:
        return search_images(query_mode, text_query, image_query, k, index_name)
    else:
        # print(k)
        return search_images_custom(query_mode, text_query, image_query, k)



def check_if_custom_index_exists():
    if os.path.exists("./index/custom_image_index.index") and os.path.exists("./index/custom_metadata.json") and len(os.listdir(direc_custom))>0:
        return gr.update("Custom Index Exists. To delete click clear button below.")

def clear_custom_image_folder():
    index_file = "./index/custom_image_index.index"
    json_file = "./index/custom_metadata.json"
    if os.path.exists(index_file):
        try:
            os.remove(index_file)
            print(f"Deleted index file: {index_file}")
        except Exception as e:
            print(f"Failed to delete index file {index_file}. Reason: {e}")

    if os.path.exists(json_file):
        try:
            os.remove(json_file)
            session.metadata.clear()
            print(f"Deleted metadata file: {json_file}")
        except Exception as e:
            print(f"Failed to delete metadata file {json_file}. Reason: {e}")

    for filename in os.listdir(direc_custom):
        file_path = os.path.join(direc_custom, filename)
        try:
            if os.path.isfile(file_path) or os.path.islink(file_path):
                os.unlink(file_path)
            elif os.path.isdir(file_path):
                shutil.rmtree(file_path)

            # session.index.clear()
            # session.index_size.clear()
            # if os.path.isfile("C:/AI_Assisstant/app/indexcustom_image_index.index"):
            print(f"Deleted file: {file_path}")
        except Exception as e:
            print(f'Failed to delete {file_path}. Reason: {e}')
    return gr.update(value = None), gr.update(value = None), gr.update(value = None), gr.update(value = None)


def clear_chatbot():
    session.chat_history.clear()
    session.metadata_selected_gpu = None
    return gr.update(value = None), gr.update(visible=False), gr.update(visible=False), gr.update(visible=False), gr.update(value = None), gr.update(value = None), gr.update(value = "Enter a query and search for a product first.")

def llm_run(messages, context_text):
    # print("Running LLM")
    # print(messages)
    # print(context_text)
    system_prompt = (
        "You are a helpful GPU customer support assistant. "
        "You can only answer questions related to the product described in the context. "
        "If the user asks about something unrelated, politely inform them that you can only answer questions about the provided product. "
        "If the user's issue is new and not in the context, suggest they contact customer support."
    )

    return session.client.chat.completions.create(
        model="gpt-4.1-mini",
        messages=[
            # {"role": "system", "content": "You are a helpful GPU customer support assistant. If the User query is not related to the product, politely inform them that you can only answer questions related to the product. If the user issue is new i.e. you can't find any relevant information in the context provided, suggest they contact customer support."},
            {"role": "system", "content": system_prompt},
            {"role": "assistant", "content": str(context_text)},
        ]+messages,
        max_tokens=500,
    )

def selected_prebuilt_index(index_name):
    session.index_size = read_index(f"./index/{index_name}").ntotal if os.path.exists(f"./index/{index_name}") else 0
    return "Using prebuilt demo index with {} images.".format(session.index_size), gr.update(value = None), gr.update(value = None, visible=False), gr.update(value = None, visible=False)


def selected_gpu_model_fn(evt: gr.SelectData, gallery_items, use_prebuilt_index):
    # evt.index = index of the selected image
    if use_prebuilt_index:
        if evt.index is not None:
            label = gallery_items[evt.index][1]  # second item in tuple = label
            with open("./index/hardware_metadata.json", "r") as f:
                metadata = json.load(f)
                session.metadata_selected_gpu = (next((item for item in metadata if item["filename"] == "_".join([w.lower().strip() for w in label.split()])+".jpg"), None))
                # print(label.split())
                # print(label+".jpg")
                # print(session.metadata_selected_gpu)
                session.chat_history.clear()
            return gr.update(visible=True, value=label), gr.update(value = None)
        return None
    return gr.update(visible=False), gr.update(visible = False)


with gr.Blocks() as demo:
    gr.Markdown("# Chat with Photo Library 💬\nTry with a demo photo index, or upload your own images and search by text/image.")
    use_prebuilt = gr.Checkbox(label="Use prebuilt demo index (faster, no upload)")
    select_prebuilt_index = gr.Dropdown(choices=["app_image_index.index", "hardware_index.index"], value="app_image_index.index", label="Select demo index", visible=False)
    file_input = gr.Files(label="Upload your photos", file_count="multiple", type="filepath")
    status = gr.Textbox(label="Status", interactive=False)
    query_mode = gr.Radio(choices=["Text", "Image"], value="Text", label="Search by")
    text_query = gr.Textbox(label="Text Query", visible=True)
    image_query = gr.File(label="Image Query", visible=False)
    k = gr.Number(label="Number of images to index (for custom index)", value=5, precision=0, visible=True)
    run_btn = gr.Button("Search")
    clear_btn = gr.Button("Clear")
    # run_btn_custom = gr.Button("Search from custom index")
    # custom_index_exists_message = gr.Textbox("Custom Index Exists. To delete click clear button below.")
    gallery = gr.Gallery(label=f"Top Results", columns=5, rows=1, interactive=False)
    selected_gpu_model = gr.Textbox(label="Selected GPU Model", visible=False)
    chatbot = gr.Chatbot(label="Customer Support" ,type="messages", visible=False)
    chat_text = gr.Textbox(label="Ask a question about the product you searched for", visible=False)
    chat_btn = gr.Button("Send", visible=False)
    clear_btn_2 = gr.Button("Clear")

    def respond(message, chat_history):
        print("Running bot")
        session.chat_history.append({"role": "user", "content": message})
        bot_message = llm_run(session.chat_history, session.metadata_selected_gpu).choices[0].message.content
        session.chat_history.append({"role": "assistant", "content": bot_message})
        time.sleep(2)
        return "", session.chat_history

    # When switching index source
    use_prebuilt.change(set_mode, use_prebuilt, [file_input, status, select_prebuilt_index, chatbot, chat_text, chat_btn, clear_btn, clear_btn_2, gallery, text_query, image_query])
    file_input.change(upload_images, file_input, status)
    demo.load(check_if_custom_index_exists, [], status)

    gallery.select(selected_gpu_model_fn,[gallery, use_prebuilt], [selected_gpu_model, chatbot])

    select_prebuilt_index.select(selected_prebuilt_index, select_prebuilt_index, [status, gallery, chatbot, chat_text])



    # Toggle search widgets
    def toggle_query(qtype):
        return gr.update(visible=qtype=="Text"), gr.update(visible=qtype=="Image")
    query_mode.change(toggle_query, query_mode, [text_query, image_query])


    run_btn.click(
        run_search,
        [query_mode, text_query, image_query, use_prebuilt, k, select_prebuilt_index],
        [gallery, status, chatbot, chat_text, chat_btn]
    )

    text_query.submit(
        run_search,
        [query_mode, text_query, image_query, use_prebuilt, k, select_prebuilt_index],
        [gallery, status, chatbot, chat_text, chat_btn]
    )

    chat_btn.click(
        respond,
        [chat_text, chatbot],
        [chat_text, chatbot]
    )

    clear_btn.click(
        clear_custom_image_folder,
        [],
        [file_input, gallery, text_query, image_query]
    )

    clear_btn_2.click(
        clear_chatbot,
        [],
        [chatbot, chat_text, selected_gpu_model, chat_btn, text_query, gallery, status]
    )


demo.launch(share=False)
