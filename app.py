import streamlit as st 

#Define path to the directory containing all picked files
#Paths to pickles models and vectorizer
#Function for text_cleaning
#Function for models path
#Function for vectorizer path
#Function for vectorizer path


st.image("anime.jpeg")

st.selectbox('Select an algorithm', ['Content Based Filtering', 'Collaborative Based Filtering'])

st.title('Enter Your Three Favorite Movies')
 
st.selectbox("Enter first movie choice", ['Comedy', 'Romance'])

st.selectbox("Enter second movie choice", ['Comedy', 'Romance'])

st.selectbox("Enter third movie choice", ['Comedy', 'Romance'])

st.button('Recommend')
