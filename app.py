import streamlit as st
import os
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.cluster import KMeans
from sklearn.datasets import make_blobs
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt

# Define path to the directory containing all pickled files
MODEL_DIR = os.path.abspath(os.path.dirname(__file__))

# Paths to pickled models 
cf_model = os.path.join(MODEL_DIR, 'cf_model.pkl')
cb_model = os.path.join(MODEL_DIR, 'cb_model.pkl')
tf_model = os.path.join(MODEL_DIR, 'tf_model.h5')

#Function of user ratings
def plot_user_rating_distribution(train_merged):
    plt.figure(figsize=(10, 6))
    sns.histplot(train_merged['user_rating'], bins=10, kde=True)
    plt.title('Distribution of User Ratings')
    plt.xlabel('User Rating')
    plt.ylabel('Frequency')
    st.pyplot(plt)

#Function of top anime ratings
def top_anime_ratings_count(df, n):
    """
    Counts the number of user ratings for the top n anime titles.
    Parameters
    ----------
        df (DataFrame): input DataFrame
        n (int): number of anime titles to show
    Returns
    -------
        None: displays a barplot of top n anime titles by the number of ratings
    """
    # Create the barplot
    plt.figure(figsize=(10, 6))
    data = df['name'].value_counts().head(n)
    
    ax = sns.barplot(x=data.index, y=data, order=data.index, palette='viridis', edgecolor="white")
    
    for p in ax.patches:
        ax.text(p.get_x() + p.get_width() / 2., p.get_height(), f'{int(p.get_height())}', fontsize=11, ha='center', va='bottom')

    plt.title(f'Top {n} Anime Titles by Number of Ratings', fontsize=14)
    plt.xlabel('Anime Title')
    plt.ylabel('Number of Ratings')
    plt.xticks(rotation=45, ha='right')

    # Display the plot in Streamlit
    st.pyplot(plt)

    # Print summary information
    st.write("Combined number of ratings for top anime titles:", data.sum())
    st.write("Total number of unique anime titles:", df['name'].nunique())

#Anime Recommendation System
def plot_avg_user_rating_per_anime(df):
    """
    Plots the average user rating for the top 10 anime titles.
    Parameters
    ----------
        df (DataFrame): input DataFrame
    Returns
    -------
        None: displays a bar plot of top 10 anime by average user rating
    """
    # Compute average user rating per anime
    avg_user_rating_per_anime = df.groupby('name')['user_rating'].mean().sort_values(ascending=False).head(10)
    
    # Create the bar plot
    plt.figure(figsize=(12, 6))
    avg_user_rating_per_anime.plot(kind='bar', color='skyblue', edgecolor='black')
    plt.title('Top 10 Anime by Average User Rating')
    plt.xlabel('Anime')
    plt.ylabel('Average User Rating')
    plt.xticks(rotation=45, ha='right')

    # Display the plot in Streamlit
    st.pyplot(plt)

#Show Genre Distribution
def plot_genre_distribution(df):
    """
    Plots the distribution of anime genres, showing the top 10 genres.
    Parameters
    ----------
        df (DataFrame): input DataFrame with a 'genre' column.
    Returns
    -------
        None: displays a bar plot of top 10 anime genres
    """
    # Compute the genre distribution
    genre_count = df['genre'].str.split(', ').explode().value_counts().head(10)
    
    # Create the bar plot
    plt.figure(figsize=(12, 6))
    genre_count.plot(kind='bar', color='lightcoral', edgecolor='black')
    plt.title('Top 10 Anime Genres')
    plt.xlabel('Genre')
    plt.ylabel('Frequency')
    plt.xticks(rotation=45, ha='right')

    # Display the plot in Streamlit
    st.pyplot(plt)

def plot_rating_distribution_by_genre(df):
    """
    Plots the distribution of user ratings by genre, showing the top 10 genres.
    Parameters
    ----------
        df (DataFrame): input DataFrame with 'genre' and 'user_rating' columns.
    Returns
    -------
        None: displays a box plot of user ratings by genre
    """
    # Get the top 10 genres
    top_genres = df['genre'].str.split(', ').explode().value_counts().head(10).index
    
    # Filter the DataFrame for these top genres
    genre_ratings = df[df['genre'].str.contains('|'.join(top_genres))]
    genre_ratings = genre_ratings.explode('genre')
    
    # Create the box plot
    plt.figure(figsize=(12, 8))
    sns.boxplot(data=genre_ratings, x='genre', y='user_rating', palette='Set2')
    plt.title('Distribution of Ratings by Genre')
    plt.xlabel('Genre')
    plt.ylabel('User Rating')
    plt.xticks(rotation=90)
    
    # Display the plot in Streamlit
    st.pyplot(plt)

def plot_distribution_of_ratings_by_genre(df):
    """
    Plots the distribution of user ratings by genre for the top 10 genres.
    
    Parameters
    ----------
    df (DataFrame): The DataFrame containing 'genre' and 'user_rating' columns.
    
    Returns
    -------
    None: Displays a box plot of user ratings by genre in the Streamlit app.
    """
    # Get the top 10 genres
    top_genres = df['genre'].str.split(', ').explode().value_counts().head(10).index
    
    # Filter the DataFrame for these top genres
    genre_ratings = df[df['genre'].str.contains('|'.join(top_genres))]
    genre_ratings = genre_ratings.explode('genre')
    
    # Create the box plot
    plt.figure(figsize=(12, 8))
    sns.boxplot(data=genre_ratings, x='genre', y='user_rating', palette='Set2')
    plt.title('Distribution of Ratings by Genre')
    plt.xlabel('Genre')
    plt.ylabel('User Rating')
    plt.xticks(rotation=90)
    
    # Display the plot in Streamlit
    st.pyplot(plt)

#Show Word Cloud of Anime Titles
def plot_word_cloud(df):
    """
    Generates and displays a word cloud of anime titles.

    Parameters
    ----------
    df (DataFrame): The DataFrame containing the 'name' column with anime titles.

    Returns
    -------
    None: Displays a word cloud in the Streamlit app.
    """
    # Combine all anime titles into a single string
    titles_text = ' '.join(df['name'])

    # Generate the word cloud
    wordcloud = WordCloud(width=800, height=400, background_color='white').generate(titles_text)

    # Create and display the word cloud plot
    plt.figure(figsize=(12, 6))
    plt.imshow(wordcloud, interpolation='bilinear')
    plt.axis('off')
    plt.title('Word Cloud of Anime Titles')

    # Display the plot in Streamlit
    st.pyplot(plt)


def plot_rating_relationship(df):
    """
    Generates and displays a scatter plot showing the relationship between anime rating and user rating.

    Parameters
    ----------
    df (DataFrame): The DataFrame containing 'anime_rating' and 'user_rating' columns.

    Returns
    -------
    None: Displays a scatter plot in the Streamlit app.
    """
    plt.figure(figsize=(10, 6))
    sns.scatterplot(data=df, x='anime_rating', y='user_rating', alpha=0.5)
    plt.title('Relationship between Anime Rating and User Rating')
    plt.xlabel('Anime Rating')
    plt.ylabel('User Rating')
    
    # Display the plot in Streamlit
    st.pyplot(plt)  


 # Streamlit Interface
st.subheader('Anime Recommendation System')

# Image
st.image("anime.jpeg")

def main():
    """Anime Recommendation App"""
    
    # Creating sidebar with selection box -
    # you can create multiple pages this way
options = ["Prediction", "Information", "Exploratory Data Analysis"]
    
    # Ensure each widget has a unique key to avoid DuplicateWidgetID error
selection = st.sidebar.selectbox("Choose Option", options)
    
   
    
# Example feature mapping for demonstration
def map_user_input(user_input_features):
    # Implement this function based on how you encode or transform your inputs
    # For simplicity, assuming direct conversion to a numpy array
    return np.array([user_input_features])

#Building out the "Informnation" page
if selection == "Information":
    st.info("General Information")
    st.markdown("""
                Discover Your Next Favorite Anime with Our Smart Recommender System!
Dive into a world where your anime watching experience is transformed with personalized recommendations! Our cutting-edge recommender system is designed to predict how much you'll love a new anime title based on your unique viewing history and preferences.

What Makes Our System Special?

Dual Filtering Techniques: We harness the power of both collaborative filtering and content-based filtering to offer you the most accurate predictions. Whether you're a fan of action-packed adventures or heartfelt dramas, our system gets you.
Tailored for You: By analyzing your historical preferences and the nuances of anime content, we craft recommendations that are as unique as your taste.
End-to-End Development: From data preprocessing and model training to real-time recommendations, we've covered every stage of machine learning development to bring you a seamless experience.
Interactive Streamlit App: Explore, interact, and discover with our user-friendly Streamlit application, designed to make finding your next favorite anime fun and easy.
Objective of Our Recommender System:
Our goal is simple – to enhance your anime watching journey by predicting how you'll rate new titles based on your viewing history. We analyze your preferences and the intricate details of anime content to offer recommendations that truly resonate with you.

Experience the future of anime recommendations today. Your next favorite anime is just a recommendation away!  
                """)


# Building out the "Exploratory Data Analysis" page
if selection == "Exploratory Data Analysis":
    st.subheader("Exploratory Data Analysis")

    st.subheader('Anime Recommendation System')

    # Assuming train_merged is already loaded and preprocessed
    # Replace the following line with your actual data loading code
    train_merged = pd.read_csv('train_merged.csv')

    if st.button('Plot User Rating Distribution'):
        plot_user_rating_distribution(train_merged)

    st.subheader('Anime Recommendation System') 

    # Assuming df is already loaded and preprocessed
    # Replace the following line with your actual data loading code
    df = pd.read_csv('path_to_your_data.csv')

    n = st.slider('Select the number of top anime titles to show', min_value=1, max_value=20, value=5)

    if st.button('Show Top Anime Ratings'):
        top_anime_ratings_count(df, n)

    st.subheader('Anime Recommendation System')

    # Load your DataFrame (replace with actual data loading)
    df = pd.read_csv('path_to_train_merged.csv')

    # Button to plot the average user rating
    if st.button('Show Average User Rating for Top 10 Anime'):
        plot_avg_user_rating_per_anime(df)

    st.subheader('Anime Recommendation System')

    # Load your DataFrame (replace with actual data loading)
    df = pd.read_csv('path_to_train_merged.csv')

    # Button to plot genre distribution
    if st.button('Show Genre Distribution'):
        plot_genre_distribution(df)

    st.subheader('Anime Recommendation System')

    # Load your DataFrame (replace with actual data loading)
    df = pd.read_csv('path_to_train_merged.csv')

    # Button to plot rating distribution by genre
    if st.button('Show Rating Distribution by Genre'):
        plot_rating_distribution_by_genre(df)

    st.subheader('Anime Recommendation System')
    
    # Load your DataFrame (replace with actual data loading)
    df = pd.read_csv('path_to_train_merged.csv')  # Update with the actual path to your dataset

    # Button to plot the word cloud
    if st.button('Show Word Cloud of Anime Titles'):
        plot_word_cloud(df)


    st.subheader('Anime Recommendation System')
    
    # Load your DataFrame (replace with actual data loading)
    df = pd.read_csv('path_to_train_merged.csv')  # Update with the actual path to your dataset

    # Button to plot the word cloud
    if st.button('Show Word Cloud of Anime Titles'):
        plot_word_cloud(df)


if selection == "Prediction":
    # Selection for algorithm selection
    algorithm = st.selectbox('Select an algorithm', ['Content Based Filtering', 'Collaborative Filtering', 'Hybrid'])

    # Inputs for user’s favorite movies
    st.header('Enter Your Three Favorite Movies')

    movie1 = st.selectbox("Enter first movie choice", ['Comedy', 'Romance', 'Action', 'Drama'])
    movie2 = st.selectbox("Enter second movie choice", ['Comedy', 'Romance', 'Action', 'Drama'])
    movie3 = st.selectbox("Enter third movie choice", ['Comedy', 'Romance', 'Action', 'Drama'])
    
    if st.button('Recommend'):
    # Check if all inputs are provided
        if not movie1 or not movie2 or not movie3:
            st.error('Please enter all three movie choices.')
        else:
            # Convert the user input to the appropriate format
            user_input_features = [movie1, movie2, movie3]
            user_features = map_user_input(user_input_features)  # Convert to model input format
        if algorithm == 'Content Based Filtering':
            # Predict using Content-Based Filtering
            content_based_prediction = cb_model.predict(user_features)  # Ensure this method matches model expectations
            st.write(f'Content-Based Prediction: {content_based_prediction}')

        elif algorithm == 'Collaborative Filtering':
            # Predict using Collaborative Filtering
            collaborative_prediction = cf_model.predict(user_features)  # Ensure this method matches model expectations
            st.write(f'Collaborative Filtering Prediction: {collaborative_prediction}')

        elif algorithm == 'Hybrid':
            # Predict using Hybrid Recommendation
            alpha = st.slider('Select weight for Collaborative Filtering', 0.0, 1.0, 0.5)
            hybrid_prediction = hybrid_recommendation(user_features, cf_model, cb_model, tf_model, alpha)
            st.write(f'Hybrid Recommendation Prediction: {hybrid_prediction}')
            
def hybrid_recommendation(user_features, cf_model, cb_model, tf_model, alpha=0.5):
    # Ensure user_features is in correct format for each model
    # Collaborative Filtering Prediction
    cf_prediction = cf_model.predict(user_features)

    # Content-Based Filtering Prediction
    cb_prediction = cb_model.predict(user_features, user_features)  # Adjust as needed for your model

    # TensorFlow Model Prediction
    user_input = np.array([user_features[0]])  # Example, adapt as needed
    anime_input = np.array([user_features[1]])  # Example, adapt as needed
    tf_prediction = tf_model.predict([user_input, anime_input])

    # Combine predictions
    hybrid_prediction = alpha * cf_prediction + (1 - alpha) * cb_prediction + (1 - alpha) * tf_prediction
    return hybrid_prediction

# Required to let Streamlit instantiate our web app.  
if __name__ == '__main__':
	main()




