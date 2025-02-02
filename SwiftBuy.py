import streamlit as st
import os
from PIL import Image
import numpy as np
import pickle
import tensorflow
from tensorflow.keras.preprocessing import image
from tensorflow.keras.layers import GlobalMaxPooling2D
from tensorflow.keras.applications.resnet50 import ResNet50, preprocess_input
from sklearn.neighbors import NearestNeighbors
from numpy.linalg import norm

# Set page config for better UI
st.set_page_config(page_title="SwiftBuy", layout='wide')

# Load feature list and filenames
feature_list = np.array(pickle.load(open('embeddings.pkl', 'rb')))
filenames = pickle.load(open('filenames.pkl', 'rb'))


# Normalize the file paths to ensure consistency across environments
def normalize_path(path):
    return path.replace('\\', '/')

# Load ResNet50 model
model = ResNet50(weights='imagenet', include_top=False, input_shape=(224, 224, 3))
model.trainable = False
model = tensorflow.keras.Sequential([
    model,
    GlobalMaxPooling2D()
])

# Custom header styling
st.markdown("""
    <h1 style='text-align: center; color: #4A90E2;'>SwiftBuy Image Recommender</h1>
    <p style='text-align: center; font-size: 18px;'>Upload an image and find visually similar products instantly!</p>
    <hr style='border:1px solid #ddd;'>
    """, unsafe_allow_html=True)

# Directory to save uploaded files
upload_dir = 'uploads'
if not os.path.exists(upload_dir):
    os.makedirs(upload_dir)

def save_uploaded_file(uploaded_file):
    """Save uploaded file to the specified directory."""
    try:
        with open(os.path.join(upload_dir, uploaded_file.name), 'wb') as f:
            f.write(uploaded_file.getbuffer())
        return True
    except Exception as e:
        st.error(f"Error saving file: {e}")
        return False

def feature_extraction(img_path, model):
    """Extract features from the image using the pre-trained model."""
    img = image.load_img(img_path, target_size=(224, 224))
    img_array = image.img_to_array(img)
    expanded_img_array = np.expand_dims(img_array, axis=0)
    preprocessed_img = preprocess_input(expanded_img_array)
    result = model.predict(preprocessed_img).flatten()
    normalized_result = result / norm(result)
    return normalized_result

def recommend(features, feature_list):
    """Recommend similar images based on extracted features."""
    neighbors = NearestNeighbors(n_neighbors=6, algorithm='brute', metric='euclidean')
    neighbors.fit(feature_list)
    distances, indices = neighbors.kneighbors([features])
    return indices

# File upload step
uploaded_file = st.file_uploader("Choose an image", type=['png', 'jpg', 'jpeg'])
if uploaded_file is not None:
    if save_uploaded_file(uploaded_file):
        col1, col2, col3 = st.columns([1, 2, 1])  # Center align image
        with col2:
            st.image(uploaded_file, caption='Uploaded Image', width=250)
        
        if st.button('Get Recommendations'):
            with st.spinner('Processing...'):
                features = feature_extraction(os.path.join(upload_dir, uploaded_file.name), model)
                indices = recommend(features, feature_list)
            
            # Display recommended images
            st.subheader("Recommended Products:")
            cols = st.columns(5)
            for i, col in enumerate(cols):
                if i < len(indices[0]):
                    with col:
                        st.image(filenames[indices[0][i]], use_container_width=True)
    else:
        st.error("Some error occurred in file upload.")
