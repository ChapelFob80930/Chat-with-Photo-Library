# Chat with Photo Library

Interact with your photo library using natural language queries powered by AI embeddings. This project uses **CLIP** embeddings to connect text and images, enabling semantic search and conversational interaction with your photo collection.

---

## 🚀 Features
- Generate **CLIP embeddings** for your photo library
- Search photos using **natural language queries** (e.g., "sunset at the beach")
- Chat-like interface to explore and interact with your images
- Supports scalable storage of embeddings for fast retrieval

---

## 📂 Project Structure
- `clip_embeddings.ipynb` &rarr; Main notebook for generating embeddings and running queries
- `images/` &rarr; Your photo library (not included in repo)
- `embeddings/` &rarr; Stores generated embeddings (can be connected to a vector database later)

---

## ⚙️ Installation

1. Clone the repository:
    ```bash
    git clone https://github.com/ChapelFob80930/Chat-with-Photo-Library.git
    cd Chat-with-Photo-Library
    ```
2. Create a virtual environment and install dependencies:
    ```bash
    pip install -r requirements.txt
    ```
3. (Optional) If using GPU with PyTorch:
    ```bash
    pip install torch torchvision --extra-index-url https://download.pytorch.org/whl/cu121
    ```

---

## 📖 Usage

1. Place your images inside the `images/` folder.
2. Open the notebook:
    ```bash
    jupyter notebook clip_embeddings.ipynb
    ```
3. Run all cells to:
    - Generate embeddings
    - Search your photos using text queries
    - Experiment with conversational queries

---

## 🔮 Roadmap
- [ ] Add a simple **web interface** with chat functionality
- [ ] Store embeddings in a **vector database** (e.g., Pinecone, FAISS)
- [ ] Support multi-user photo libraries
- [ ] Deploy as a **web app** with FastAPI + Next.js

---

## 🤝 Contributing
Pull requests and suggestions are welcome!

---

## 📜 License
This project is licensed under the MIT License.
