# imports the Pandas library, used for data manipulation and analysis.
import pandas as pd

import numpy as np  # useful for numerical computations.

# imports the Natural Language Toolkit (NLTK) for text processing tasks like tokenization and stemming.
import nltk

# Imports the Snowball stemmer, a tool for reducing words to their root form.
from nltk.stem.snowball import SnowballStemmer

# imports the TF-IDF vectorizer for converting text data into numerical form for similarity calculations.
from sklearn.feature_extraction.text import TfidfVectorizer

# Imports the cosine similarity function for comparing text documents based on their vector representation.
from sklearn.metrics.pairwise import cosine_similarity

# Imports Streamlit, a library for building interactive web apps.
import streamlit as st

# Imports the Python Imaging Library (PIL) for image handling.
from PIL import Image

# Download the required NLTK data used for splitting text into tokens (words or sentences).
nltk.download('punkt_tab')

# Load the dataset with caching to improve performance
@st.cache_data
# function to load the dataset.
def load_data():
    data = pd.read_csv('Ecommerce_product.csv')
    data = data.drop('id', axis=1)
    return data  # Returns the preprocessed dataset.

# Define tokenizer and stemmer
stemmer = SnowballStemmer('english')  # Initializes the Snowball stemmer for the English language.

def tokenize_and_stem(text):
    # Tokenizes the text into words after converting it to lowercase.
    tokens = nltk.word_tokenize(text.lower())
    # Applies stemming to each token to reduce it to its root form.
    stems = [stemmer.stem(t) for t in tokens]
    return stems

# Create stemmed tokens column
# defines a function to add a new column of stemmed tokens to the dataset.
def create_stemmed_tokens_column(data):
    # Combines the Title and Description of each row, tokenizes and stems them, 
    # and stores the result in a new column called stemmed_tokens.
    data['stemmed_tokens'] = data.apply(
        lambda row: tokenize_and_stem(row['Title'] + ' ' + row['Description']), axis=1
    )
    # Returns the updated dataset:
    return data

# -------- Define TF-IDF vectorizer and cosine similarity function

# initializes the TF-IDF vectorizer, specifying tokenize_and_stem as the tokenizer.
tfidf_vectorizer = TfidfVectorizer(tokenizer=tokenize_and_stem, token_pattern=None)

# Defines a function to compute cosine similarity between two sets of text tokens.
def cosine_sim(text1, text2):
    # Joins the tokens in text1,text2 into a single string.
    text1_concatenated = ' '.join(text1)
    text2_concatenated = ' '.join(text2)
    # Computes the TF-IDF vectors for the two input texts.
    tfidf_matrix = tfidf_vectorizer.fit_transform([text1_concatenated, text2_concatenated])
    # ----- Returns the cosine similarity score between the two vectors.
    return cosine_similarity(tfidf_matrix)[0][1]

# Define search function for products based on a query.
def search_products(query, data):
    # Tokenizes and stems the user’s query.
    query_stemmed = tokenize_and_stem(query)
    # Computes the similarity of the query with each product’s stemmed tokens.
    data['similarity'] = data['stemmed_tokens'].apply(lambda x: cosine_sim(query_stemmed, x))
    # Sorts the dataset by similarity in descending order and selects the top 10 results.
    results = data.sort_values(by=['similarity'], ascending=False).head(10)
    
    # Checks if no matching products are found.
    if results.empty:
        return None
    # Returns a subset of columns from the top results.
    return results[['Title', 'Description', 'Category', 'similarity']]

# Main function to run the Streamlit app
def main():
    # Load the image and display it
    img = Image.open('swift.png')
    
    # Make the image responsive for mobile
    st.image(img, width=st.columns([3, 1])[0].width * 0.8)  # Adjust width based on screen size
    st.title("Search Engine and Product Recommendation System")

    # Load and preprocess the data
    data = load_data()
    data = create_stemmed_tokens_column(data)

    # ----------- User input and search functionality ---------
    
    # Create a 2-column layout for input field and button
    col1, col2 = st.columns([3, 1])

    with col1:
        # Creates a text input field for the user to enter a query with a mobile-friendly width.
        query = st.text_input("Enter Product Name", placeholder="Search for a product...", max_chars=50)

    with col2:
        # Adds a mobile-friendly search button.
        submit = st.button('Search', use_container_width=True)

    # Checks if the search button is clicked.
    if submit:
        if query:
            # Searches for matching products.
            res = search_products(query, data)
            # Checks if any results are found.
            if res is not None:
                st.write(res)
            else:
                # Displays a message if no matches are found.
                st.write("No matching products found. Please try a different query.")
        else: 
            # Handles cases where the query is empty.
            st.write("Please enter a product name to search.")

# Ensures the script runs only when executed directly.
if __name__ == "__main__":
    # Calls the main function to run the app.
    main()
