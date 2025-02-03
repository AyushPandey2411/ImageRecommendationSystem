import streamlit as st
import os
from PIL import Image
import numpy as np
import pickle
import tensorflow as tf
from tensorflow.keras.preprocessing import image
from tensorflow.keras.layers import GlobalMaxPooling2D
from tensorflow.keras.applications.resnet50 import ResNet50, preprocess_input
from sklearn.neighbors import NearestNeighbors
from numpy.linalg import norm

# Load feature list and filenames
feature_list = np.array(pickle.load(open('embeddings.pkl', 'rb')))
filenames = pickle.load(open('filenames.pkl', 'rb'))

# Normalize file paths for cross-platform compatibility
def normalize_path(path):
    return path.replace('\\', '/')

# Load ResNet50 model
model = ResNet50(weights='imagenet', include_top=False, input_shape=(224, 224, 3))
model.trainable = False
model = tf.keras.Sequential([
    model,
    GlobalMaxPooling2D()
])

# Set up Streamlit page config for mobile responsiveness
st.set_page_config(page_title='SwiftBuy Recommender', layout='wide')

# Display logo and title
st.image('swift.png', width=400)
st.title('SwiftBuy Image Recommender System')

# Directory to save uploaded files
upload_dir = 'uploads'
os.makedirs(upload_dir, exist_ok=True)

def save_uploaded_file(uploaded_file):
    """Save uploaded file to the specified directory."""
    file_path = os.path.join(upload_dir, uploaded_file.name)
    with open(file_path, 'wb') as f:
        f.write(uploaded_file.getbuffer())
    return file_path

def feature_extraction(img_path, model):
    """Extract features from the image using the pre-trained model."""
    img = image.load_img(img_path, target_size=(224, 224))
    img_array = image.img_to_array(img)
    expanded_img_array = np.expand_dims(img_array, axis=0)
    preprocessed_img = preprocess_input(expanded_img_array)
    result = model.predict(preprocessed_img).flatten()
    return result / norm(result)

def recommend(features, feature_list):
    """Recommend similar images based on extracted features."""
    neighbors = NearestNeighbors(n_neighbors=6, algorithm='brute', metric='euclidean')
    neighbors.fit(feature_list)
    distances, indices = neighbors.kneighbors([features])
    return indices

# File upload step
uploaded_file = st.file_uploader("Choose an image", type=['png', 'jpg', 'jpeg'])
if uploaded_file is not None:
    file_path = save_uploaded_file(uploaded_file)
    display_image = Image.open(file_path)
    st.image(display_image, caption='Uploaded Image', width=300)
    
    # Extract features and get recommendations
    if st.button('Get Recommendations'):
        with st.spinner('Processing image...'):
            features = feature_extraction(file_path, model)
            indices = recommend(features, feature_list)
            
        st.subheader("Recommended Images:")
        cols = st.columns(5)
        for i, col in enumerate(cols):
            if i < len(indices[0]):
                with col:
                    recommended_image_path = normalize_path(filenames[indices[0][i]])
                    try:
                        recommended_image = Image.open(recommended_image_path)
                        st.image(recommended_image, use_container_width=True)
                    except FileNotFoundError:
                        st.warning(f"Image not found: {recommended_image_path}")
