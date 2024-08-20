import streamlit as st
import pandas as pd
import pickle

with open('knn_model_anime.pkl', 'rb') as file:
    knn = pickle.load(file)

with open('scaler_anime.pkl', 'rb') as file:
    scaler = pickle.load(file)

ratings_df = pd.read_csv('users-score-2023.csv')
anime_df = pd.read_csv('anime-list.csv')
ratings_df = ratings_df.head(15000)

user_anime_matrix = ratings_df.pivot_table(index='user_id', columns='anime_id', values='rating')
user_anime_matrix = user_anime_matrix.fillna(0)

normalized_matrix = scaler.transform(user_anime_matrix)

def recommend_anime(user_id, user_anime_matrix, knn, n_recommendations=10):
    user_idx = user_anime_matrix.index.get_loc(user_id)
    distances, indices = knn.kneighbors([normalized_matrix[user_idx]])
    similar_users = indices.flatten()
    similar_users_ratings = user_anime_matrix.iloc[similar_users]
    mean_ratings = similar_users_ratings.mean(axis=0)
    user_ratings = user_anime_matrix.loc[user_id]
    mean_ratings = mean_ratings[user_ratings == 0]
    recommendations = mean_ratings.sort_values(ascending=False)
    recommended_anime_ids = recommendations.index[:n_recommendations]
    recommended_anime = anime_df[anime_df['anime_id'].isin(recommended_anime_ids)]
    return recommended_anime

st.title('Anime Recommendation System')
user_id = st.number_input('Enter user ID:', min_value=1, max_value=len(user_anime_matrix), value=1)

if st.button('Get Recommendations'):
    recommendations = recommend_anime(user_id=user_id, user_anime_matrix=user_anime_matrix, knn=knn)
    st.write(recommendations[['Name', 'Genres']])
