import streamlit as st
import pandas as pd
import pickle
import tensorflow as tf
from keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.models import load_model
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import ast
import re
import string
import json
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
import nltk

nltk.download('stopwords')
stop_words = set(stopwords.words('english'))
stemmer = PorterStemmer()

st.set_page_config(page_title="Netflix Emotion Recommender", page_icon="🎬")

with open("config.json", "r") as f:
    config = json.load(f)
max_len_padding = config["max_length"]

with open('tokenizer.pkl', 'rb') as f:
    tokenizer = pickle.load(f)

with open('label_encoder.pkl', 'rb') as f:
    label_encoder = pickle.load(f)

@st.cache_resource
def load_emotion_model():
    return load_model('lstm_emotion_model.h5')

emotion_model = load_emotion_model()

emotion_to_genres = {
    'sadness': ['drama', 'family'],
    'joy': ['comedy', 'romance', 'drama', 'adventure', 'animation', 'fantasy'],
    'anger': ['action', 'adventure', 'thriller'],
    'love': ['romance', 'fantasy', 'musical'],
    'fear': ['horror', 'thriller', 'crime'],
    'surprise': ['adventure', 'mystery', 'sci-fi']
}

def preprocess(text):
    text = text.lower()
    text = re.sub(r"http\S+|@\S+|#\S+|[^a-zA-Z\s]", "", text)
    tokens = text.split()
    tokens = [stemmer.stem(word) for word in tokens if word not in stop_words]
    return " ".join(tokens)

def pred_emotion(text):
    clean_text = preprocess(text)
    sequence = tokenizer.texts_to_sequences([clean_text])
    padded = pad_sequences(sequence, maxlen=max_len_padding, padding='post')
    prediction = emotion_model.predict(padded, verbose=0)
    predicted_class = prediction.argmax(axis=1)[0]
    emotion_map = {
        0: 'sadness', 1: 'joy', 2: 'love', 3: 'anger', 4: 'fear', 5: 'surprise'
    }
    return emotion_map[predicted_class]

@st.cache_data
def load_movie_data():
    df = pd.read_csv('Netflix_Movies.csv')
    if not isinstance(df['Genres'].iloc[0], list):
        df['Genres'] = df['Genres'].apply(lambda x: [g.lower().replace(" ", "") for g in ast.literal_eval(x)])
    return df

movie_df = load_movie_data()

def recommend(emotion_label, df, emotion_to_genre, top_n=5):
    genres = emotion_to_genre.get(emotion_label.lower(), [])
    if not genres:
        return pd.DataFrame(columns=['Title'])

    target = [g.lower().replace(" ", "") for g in genres]
    filtered_df = df[df['Genres'].apply(lambda g_list: any(g in g_list for g in target))]

    if filtered_df.empty:
        return pd.DataFrame(columns=['Title'])

    cv = CountVectorizer(max_features=5000, stop_words='english')
    vector = cv.fit_transform(filtered_df['tags']).toarray()
    user_vec = cv.transform([emotion_label.lower()]).toarray()
    similarity = cosine_similarity(user_vec, vector).flatten()
    top_indices = similarity.argsort()[-top_n:][::-1]

    return filtered_df.iloc[top_indices][['Title', 'Genres', 'Imdb Score']]

st.title("🎬 Netflix Emotion-Based Recommender")
st.markdown("Tell us how you're feeling, and we'll suggest movies to match your mood!")

user_input = st.text_input("🗣️ How are you feeling right now?")

st.markdown("👇 Or pick an emotion:")
selected_emotion = st.selectbox("Select an emotion (optional)", [""] + list(emotion_to_genres.keys()))

final_emotion = None

if st.button("🎯 Recommend Now"):
    if user_input:
        with st.spinner("Detecting emotion..."):
            final_emotion = pred_emotion(user_input)
            st.success(f"Detected Emotion: **{final_emotion.capitalize()}**")
    elif selected_emotion:
        final_emotion = selected_emotion
        st.info(f"Using Selected Emotion: **{final_emotion.capitalize()}**")
    else:
        st.warning("Please enter an emotion or select one from the list above.")

   
    if final_emotion:
        with st.spinner("Finding the best matches..."):
            recommended_movies = recommend(final_emotion, movie_df, emotion_to_genres, top_n=5)

        if recommended_movies.empty:
            st.error("😔 Sorry, no movies found for this emotion.")
        else:
            st.subheader("🎥 Recommended Movies:")
            for _, row in recommended_movies.iterrows():
                st.markdown(f"""
                ---
                **{row['Title']}**  
                🎭 *Genres:* {', '.join(row['Genres'])}  
                ⭐ *IMDb Score:* {row['Imdb Score']}
                """)

if st.button("🔄 Reset"):
    st.rerun()

with st.expander("ℹ️ About this app"):
    st.markdown("""
    - Detects your emotion using an LSTM model  
    - Recommends Netflix movies based on mood-to-genre mapping  
    - Built using Streamlit, TensorFlow, and Scikit-learn  
    """)
