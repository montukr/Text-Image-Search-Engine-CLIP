import streamlit as st
import chromadb
from chromadb.utils.embedding_functions import OpenCLIPEmbeddingFunction
from chromadb.utils.data_loaders import ImageLoader
import os

db_path = "./image_vecs_db/"  # DB path

# Initialize Chroma DB client, embedding function, and data loader
client = chromadb.PersistentClient(path=db_path)
embedding_function = OpenCLIPEmbeddingFunction()
data_loader = ImageLoader()

collection = client.get_or_create_collection(
    name='multimodal_collection',
    embedding_function=embedding_function,
    data_loader=data_loader
)

# CSS styles
st.markdown("""
    <style>
        /* Center everything */
        .block-container {
            display: flex;
            flex-direction: column;
            align-items: center;
            justify-content: center;
            text-align: center;
        }
        
        /* Style buttons */
        .stButton > button {
            background-color: #007BFF !important; /* Blue */
            color: white !important;
            border-radius: 10px !important;
            padding: 10px 20px !important;
            font-size: 16px !important;
            border: none !important;
            transition: 0.3s ease;
        }
        .stButton > button:hover {
            background-color: #0056b3 !important;
        }

        /* Image borders */
        img {
            border: 2px solid rgba(255, 255, 255, 0.5) !important;
            border-radius: 8px !important;
            box-shadow: 0px 3px 6px rgba(0, 0, 0, 0.2) !important;
        }
    </style>
""", unsafe_allow_html=True)

# Title
st.markdown("<h2 style='text-align: center;'>🔍 Image Search Engine</h2>", unsafe_allow_html=True)

# Search Form (allows "Enter" to trigger search)
with st.form("search_form"):
    query = st.text_input("Enter your search query:", key="search_query")
    submitted = st.form_submit_button("Search")

if submitted and query:
    parent_path = "./images/"  # Image folder path
    results = collection.query(query_texts=[query], n_results=5, include=["distances"])

    # Display images in a grid
    st.markdown("### 🖼️ Search Results")
    c1, c2, c3, c4, c5 = st.columns(5)

    for idx, (image_id, distance) in enumerate(zip(results['ids'][0], results['distances'][0])):
        image_path = os.path.join(parent_path, image_id)
        if idx == 0:
            c1.image(image_path, caption=os.path.basename(image_path), use_container_width=True)
        elif idx == 1:
            c2.image(image_path, caption=os.path.basename(image_path), use_container_width=True)
        elif idx == 2:
            c3.image(image_path, caption=os.path.basename(image_path), use_container_width=True)
        elif idx == 3:
            c4.image(image_path, caption=os.path.basename(image_path), use_container_width=True)
        elif idx == 4:
            c5.image(image_path, caption=os.path.basename(image_path), use_container_width=True)
