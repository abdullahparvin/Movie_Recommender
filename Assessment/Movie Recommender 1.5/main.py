import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.cluster import KMeans
from sklearn.metrics.pairwise import cosine_similarity

# Load each data set (users, movies, and ratings).
users_cols = ['user_id', 'age', 'sex', 'occupation', 'zip_code']
users = pd.read_csv('u.user', sep='|', names=users_cols, encoding='latin-1')

ratings_cols = ['user_id', 'movie_id', 'rating', 'unix_timestamp']
ratings = pd.read_csv('u.data', sep='\t', names=ratings_cols, encoding='latin-1')
# The movies file contains a binary feature for each genre.

genre_cols = [
    "genre_unknown", "Action", "Adventure", "Animation", "Children", "Comedy",
    "Crime", "Documentary", "Drama", "Fantasy", "Film-Noir", "Horror",
    "Musical", "Mystery", "Romance", "Sci-Fi", "Thriller", "War", "Western"
]

movies_cols = ['movie_id', 'title', 'release_date', "video_release_date", "imdb_url"] + genre_cols
movies = pd.read_csv('u.item', sep='|', names=movies_cols, encoding='latin-1')

# Since the ids in the dataset start at 1, we shift them to start at 0.
users["user_id"] = users["user_id"].apply(lambda x: str(x - 1))
movies["movie_id"] = movies["movie_id"].apply(lambda x: str(x - 1))
movies["year"] = movies['release_date'].apply(lambda x: str(x).split('-')[-1])
ratings["movie_id"] = ratings["movie_id"].apply(lambda x: str(x - 1))
ratings["user_id"] = ratings["user_id"].apply(lambda x: str(x - 1))
ratings["rating"] = ratings["rating"].apply(lambda x: float(x))

label_encoder = LabelEncoder()

# Encoding categorical features, so they can be used by the KMeans algorithm
users['sex'] = label_encoder.fit_transform(users['sex'])
users['occupation'] = label_encoder.fit_transform(users['occupation'])


# Function which returns the title of a movie using the movie id
def get_movie(movie_id):
    name = (movies.loc[movies['movie_id'] == movie_id])['title'].iloc[0]
    return name


# Function which returns the cluster a given user belongs to
def get_user_cluster(input_user):
    cluster = input_user['cluster'].iloc[0]
    return cluster


# Compute the cosine similarity between two users
def compute_cosine_similarity(ratings, user1, user2):
    # Storing the users' ids
    user1_id = user1['user_id'].iloc[0]
    user2_id = user2['user_id']

    # Creating dataframes with only the relevant info for each user from the ratings dataset
    user1_movies_data = ratings.loc[ratings['user_id'] == user1_id]
    user2_movies_data = ratings.loc[ratings['user_id'] == user2_id]

    # Dataframes with only the movies each user has watched and the rating given
    user1_movie_ratings = user1_movies_data.loc[:, user1_movies_data.columns.drop(['user_id', 'unix_timestamp'])]
    user2_movie_ratings = user2_movies_data.loc[:, user2_movies_data.columns.drop(['user_id', 'unix_timestamp'])]

    # Merging dataframes to find the movies the users have in common and the corresponding rating.
    # Has 3 columns: movie_id, rating_x, rating_y.
    common_movies = pd.merge(user1_movie_ratings, user2_movie_ratings, how='inner', on='movie_id')

    # If users have no shared movies there is no similarity
    if len(common_movies) == 0:
        return 0

    # Arrays with each user's ratings on the common movies
    user1_ratings = np.array(common_movies['rating_x']).reshape(1, -1)
    user2_ratings = np.array(common_movies['rating_y']).reshape(1, -1)

    # Calculating the cosine similarity between the users ratings
    cos_sim = cosine_similarity(user1_ratings, user2_ratings)

    return cos_sim


# Get movie recommendation for the input user using clustering and cosine similarity
def get_recommendation(input_user, cluster):
    # Getting the movie ids of the movies the input user has watched from the ratings data
    input_id = input_user['user_id'].iloc[0]
    input_user_data = (ratings.loc[ratings['user_id'] == input_id])
    input_user_movies = input_user_data['movie_id']

    overall_scores = {}
    similarity_scores = {}

    # Using cosine similarity to determine the similarity between the target user
    # and the other users in its cluster
    for user in cluster:
        if user.equals(input_user):
            continue

        similarity_score = compute_cosine_similarity(ratings, input_user, user)

        if similarity_score <= 0:
            continue

        # Getting the movie ids of the movies that the user we are comparing with has watched
        user_id = user['user_id']
        user_data = (ratings.loc[ratings['user_id'] == user_id])
        user_movies = user_data['movie_id']

        # Finding the movies the input user has not watched
        filtered_list = []
        for movie in user_movies.to_numpy():
            if movie not in input_user_movies.to_numpy():
                filtered_list.append(movie)

        # Update the rating for each movie the input user hasn't watched based on the similarity with the user
        for item in filtered_list:
            overall_scores.update({item: (user_data.loc[user_data['movie_id'] == item, ['rating']]).to_numpy() * similarity_score})
            similarity_scores.update({item: similarity_score})

    if len(overall_scores) == 0:
        return ['No recommendations possible']

    # Ranking the movies by normalizing the scores
    movie_scores = np.array([[score / similarity_scores[item], item]
                             for item, score in overall_scores.items()], dtype=object)

    # Sorting in decreasing order
    movie_scores = movie_scores[np.argsort(movie_scores[:, 0])[::-1]]
    print(len(movie_scores))

    # Extracting the recommendations
    movie_recommendations = [movie for _, movie in movie_scores]

    return movie_recommendations


if __name__ == '__main__':
    # Removing the user id and zip code columns from the user data we will be using in the KMeans algorithm
    # They are not necessary and introduce problems in clustering
    users_encoded = users.loc[:, users.columns.drop(['user_id', 'zip_code'])]

    kmeans = KMeans()

    # Fitting the data to the KMeans model to create clusters
    users_clustered = kmeans.fit_predict(users_encoded)

    # Adding a column to the users dataframe which holds the cluster label for the each user
    users['cluster'] = users_clustered

    # Selecting random users from users data set
    rand_user = users.sample()

    # Getting the cluster that the randomly chosen user belongs to
    target_cluster = get_user_cluster(rand_user)

    # Finding all the users that also belong to the same cluster as the randomly chosen user
    cluster_members = []
    for i, user in users.iterrows():
        if user['cluster'] == target_cluster:
            cluster_members.append(user)

    # Getting movie recommendations and printing the top 10 out
    # Disclaimer: program takes a while to run
    print("Chosen user:")
    print(rand_user)
    print("\nMovie recommendations for user number " + rand_user['user_id'].iloc[0] + " :")

    movies_rec = get_recommendation(rand_user, cluster_members)
    limit = 10
    for i, movie in enumerate(movies_rec):
        if i == limit:
            break
        print(str(i + 1) + '. ' + get_movie(movie))
