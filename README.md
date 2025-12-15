# SwiftBuy - AI Powered Image Recommender System

## 🚀 Live Demo

Explore the live demo of the SwiftBuy Image Recommender System by clicking the link below:

[**SwiftBuy Live Demo**](https://swiftbuyimagerecommender.streamlit.app/)

## ✨ Overview

SwiftBuy is an AI-powered image recommendation system designed to recommend similar images based on the visual content of a user's uploaded image. It leverages **ResNet50**, a pre-trained convolutional neural network (CNN), to extract high-level features from the images. These features are used to calculate similarity scores and recommend the most similar images from a given dataset.

The system uses **nearest neighbors** algorithm to find the closest matches based on the image embeddings. This project is powered by **Streamlit** for the user interface, making it easy to upload images and view recommended results in real time.

## 🔧 Tech Stack

- **Backend**: 
  - **TensorFlow** (for model loading and inference)
  - **Keras** (for pre-trained ResNet50 model)
  - **NumPy** (for data manipulation)
  - **scikit-learn** (for nearest neighbors algorithm)
  - **Pickle** (for saving and loading image embeddings and filenames)
  - **OpenCV** (for handling image processing)

- **Frontend**:
  - **Streamlit** (for creating the interactive web interface)

## 🛠️ Features

- **Image Upload**: Users can upload an image (PNG, JPG, JPEG) to the system.
- **Recommendation Engine**: Once an image is uploaded, the system will find the top 5 most similar images from the dataset based on visual similarity.
- **Responsive Design**: The system is designed to provide an intuitive and easy-to-use interface.
- **Loading Spinner**: A loading spinner is displayed while recommendations are being processed.
- **Error Handling**: Provides error messages if the image is not found or if there’s an issue in uploading.

## 📥 Installation

To set up this project locally, follow these steps:

1. Clone the repository:

```bash
git clone https://github.com/your-username/swiftbuy-image-recommender.git
cd swiftbuy-image-recommender
