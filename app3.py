from flask import Flask, request, render_template
import pandas as pd
import numpy as np
import os

app = Flask(__name__)

# Load the top 100 popular movies
top_100_df = pd.read_csv("top_100_movies.csv")

# Pre-loaded similarity matrix and movies (adjust according to your setup)
S_df = pd.read_csv("S_top100.csv", index_col=0)
movies = pd.read_csv("ml-1m/movies.dat", sep='::', engine='python', header=None, encoding='ISO-8859-1')
movies.columns = ['MovieID', 'Title', 'Genres']
movies['mID'] = 'm' + movies['MovieID'].astype(str)

# We assume top_100_df has columns: MovieID, Title, Genres, avg_rating, etc.
# Ensure 'mID' present in top_100_df:
top_100_df['mID'] = 'm' + top_100_df['MovieID'].astype(str)

def get_image_url(m_id):
    return f"https://liangfgithub.github.io/MovieImages/{m_id[1:]}.jpg"

def myIBCF(newuser, similarity_df, top_n=10):
    # Basic item-based CF logic as before
    if not isinstance(newuser, pd.Series):
        newuser = pd.Series(newuser, index=similarity_df.columns)

    # If fewer than 2 ratings given, return top movies by popularity
    rated_count = newuser.count()
    if rated_count < 2:
        popular_movies = top_100_df.head(top_n)
        return popular_movies['mID'].tolist()

    # Otherwise, standard prediction
    rated_movies = newuser.dropna().index
    unrated_movies = newuser.index[newuser.isna()]
    predictions = pd.Series(np.nan, index=unrated_movies)

    for movie_i in unrated_movies:
        sims = similarity_df.loc[movie_i, :]
        neighbors = sims.dropna().index
        rated_neighbors = [m for m in neighbors if m in rated_movies]
        if len(rated_neighbors) == 0:
            continue
        s_ij = sims.loc[rated_neighbors].values
        w_j = newuser.loc[rated_neighbors].values
        denom = np.sum(s_ij)
        if denom == 0:
            continue
        predicted_rating = np.sum(s_ij * w_j) / denom
        predictions[movie_i] = predicted_rating

    predictions_sorted = predictions.dropna().sort_values(ascending=False)
    if len(predictions_sorted) < top_n:
        # fallback fill from top_100_df if needed
        recommended = list(predictions_sorted.index)
        already_considered = set(rated_movies).union(set(recommended))
        for pm in top_100_df['mID']:
            if pm not in already_considered:
                recommended.append(pm)
                if len(recommended) == top_n:
                    break
        return recommended
    else:
        return predictions_sorted.index[:top_n].tolist()

@app.route('/', methods=['GET'])
def index():
    # Show first 15 movies with rating options
    initial_movies = top_100_df.head(15)
    movie_data = []
    for _, row in initial_movies.iterrows():
        movie_data.append({
            "mID": row['mID'],
            "Title": row['Title'],
            "ImageURL": get_image_url(row['mID'])
        })
    return render_template("rating.html", movies=movie_data, start_index=15)

@app.route('/load_more', methods=['POST'])
def load_more():
    start_index = int(request.form.get('start_index', 15))
    next_index = start_index + 15
    more_movies = top_100_df.iloc[start_index:next_index]
    movie_data = []
    for _, row in more_movies.iterrows():
        movie_data.append({
            "mID": row['mID'],
            "Title": row['Title'],
            "ImageURL": get_image_url(row['mID'])
        })
    return render_template("rating.html", movies=movie_data, start_index=next_index)

@app.route('/recommend', methods=['POST'])
def recommend():
    # Parse user ratings from form
    user_ratings = pd.Series(np.nan, index=S_df.columns)
    for m in S_df.columns:
        val = request.form.get(m, 'N/A')
        if val != 'N/A':
            try:
                rating = float(val)
                if 1 <= rating <= 5:
                    user_ratings[m] = rating
            except:
                pass

    # Compute recommendations
    rec_ids = myIBCF(user_ratings, S_df)
    recommended_info = movies[movies['mID'].isin(rec_ids)]

    rec_data = []
    for _, row in recommended_info.iterrows():
        rec_data.append({
            "MovieID": row['mID'],
            "Title": row['Title'],
            "ImageURL": get_image_url(row['mID'])
        })

    return render_template("results.html", recommendations=rec_data)

if __name__ == '__main__':
    app.run(debug=True)
