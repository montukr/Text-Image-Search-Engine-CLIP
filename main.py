from PIL import Image
import chromadb
from chromadb.utils.embedding_functions import OpenCLIPEmbeddingFunction
from chromadb.utils.data_loaders import ImageLoader
import numpy as np
from tqdm import tqdm
import os

db_path= "./image_vecs_db/" #db path 
# Initialize Chroma DB client, embedding function, and data loader

client = chromadb.PersistentClient(path=db_path)
embedding_function = OpenCLIPEmbeddingFunction()
data_loader = ImageLoader()

collection = client.get_or_create_collection(
    name='multimodal_collection',
    embedding_function=embedding_function,
    data_loader=data_loader
)

def add_images_to_collection(folder_path):
    image_files = [os.path.join(folder_path, image_name) for image_name in os.listdir(folder_path)
                   if os.path.isfile(os.path.join(folder_path, image_name)) and image_name.lower().endswith(('.png', '.jpg', '.jpeg'))]

    for image_path in tqdm(image_files,desc="Creating Image Embeddings and Adding to DB"):
        try:
            image = np.array(Image.open(image_path))
            collection.add(
                ids=[os.path.basename(image_path)],
                images=[image]
            )
        except Exception as e:
            print(f"Error processing {image_path}: {e}")

image_folder_path= "./images/" #folder path

add_images_to_collection(image_folder_path)



# ===================================================
# Fix for NumPy 2.0 issue in ChromaDB:
# If you get an AttributeError for `np.float_`, edit:
# my_env\Lib\site-packages\chromadb\api\types.py  
# Replace `np.float_` with `np.float64` in line 102
# Or downgrade NumPy: `pip install numpy==1.26.4`
