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

# Load feature list and filenames
feature_list = np.array(pickle.load(open('embeddings.pkl', 'rb')))
filenames = pickle.load(open('filenames.pkl', 'rb'))

# Normalize file paths for compatibility
def normalize_path(path):
    return path.replace('\\', '/')

# Load ResNet50 model
model = ResNet50(weights='imagenet', include_top=False, input_shape=(224, 224, 3))
model.trainable = False
model = tensorflow.keras.Sequential([
    model,
    GlobalMaxPooling2D()
])

# UI Customization
st.set_page_config(page_title="SwiftBuy Image Recommender", layout="wide")

# Apply custom CSS
st.markdown("""
    <style>
        .main {background-color: #f5f5f5;}
        .stButton>button {border-radius: 10px; background-color: #ff5733; color: white; font-size: 16px; padding: 10px 20px;}
        .uploaded-img {border: 2px solid #ff5733; padding: 10px; border-radius: 10px;}
        .recommended-img {border: 2px solid #007BFF; padding: 5px; border-radius: 10px;}
    </style>
""", unsafe_allow_html=True)

# Banner Image
st.image("swift.png", width=500)
st.markdown("<h1 style='text-align: center; color: #FF5733; border-bottom: 4px solid #FF5733; padding-bottom: 10px;'>SwiftBuy Image Recommender System</h1>", unsafe_allow_html=True)

# Directory to save uploaded files
upload_dir = 'uploads'
if not os.path.exists(upload_dir):
    os.makedirs(upload_dir)

def save_uploaded_file(uploaded_file):
    """Save uploaded file to the specified directory."""
    try:
        file_path = os.path.join(upload_dir, uploaded_file.name)
        with open(file_path, 'wb') as f:
            f.write(uploaded_file.getbuffer())
        return file_path  # Return the path of the saved file
    except Exception as e:
        st.error(f"Error saving file: {e}")
        return None

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

# File Upload Section
st.markdown("<h3 style='text-align: center; color: #4CAF50;'>Upload an Image to Find Similar Products</h3>", unsafe_allow_html=True)

uploaded_file = st.file_uploader("", type=['png', 'jpg', 'jpeg'])
if uploaded_file is not None:
    file_path = save_uploaded_file(uploaded_file)
    if file_path:
        # Display uploaded image with styling
        st.markdown("<div class='uploaded-img'>", unsafe_allow_html=True)
        st.image(Image.open(file_path), caption='Uploaded Image', width=350)
        st.markdown("</div>", unsafe_allow_html=True)

        # Get recommendations button
        if st.button('Get Recommendations', help="Click to find similar images"):
            features = feature_extraction(file_path, model)
            indices = recommend(features, feature_list)
            
            # Display recommended images in a responsive layout
            st.markdown("<h3 style='color: #007BFF; border-bottom: 3px solid #007BFF; padding-bottom: 5px;'>Recommended Images:</h3>", unsafe_allow_html=True)
            cols = st.columns([1,1,1,1,1])
            for i, col in enumerate(cols):
                if i < len(indices[0]):
                    with col:
                        recommended_image_path = normalize_path(filenames[indices[0][i]])
                        try:
                            recommended_image = Image.open(recommended_image_path)
                            st.markdown("<div class='recommended-img'>", unsafe_allow_html=True)
                            st.image(recommended_image, use_container_width=True)
                            st.markdown("</div>", unsafe_allow_html=True)
                        except FileNotFoundError:
                            st.warning(f"Image not found: {recommended_image_path}")
    else:
        st.error("Some error occurred in file upload")
