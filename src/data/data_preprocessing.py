import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import time
import torch
from tqdm import tqdm

from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
from sklearn.metrics.pairwise import cosine_similarity

def analyze_movie_views(df, step=10):
    """
    Analyze movie view counts and generate a plot and table.

    Parameters:
    - df (pd.DataFrame): The input DataFrame with columns 'userId' and 'movieId'.
    - step (int): The step size for percentage intervals (default is 10).

    Returns:
    - pd.DataFrame: A DataFrame showing the number of movies viewed by each percentage interval.
    """
    # Calculate the total number of unique users
    num_users = df['userId'].nunique()

    # Calculate the view count for each movie
    movie_view_counts = df['movieId'].value_counts()

    # Calculate the maximum view count (the most viewed movie)
    max_views = movie_view_counts.max()

    # Initialize a dictionary to store the results
    results = {'Percentage of Users Viewing': [], 'Number of Movies': []}

    # Analyze the movie view counts at different percentage intervals
    for percentage in range(0, 101, step):
        threshold = max_views * (percentage / 100)
        num_movies = (movie_view_counts >= threshold).sum()
        real_percentage = (threshold / num_users) * 100  # Convert to percentage of total users
        results['Percentage of Users Viewing'].append(real_percentage)
        results['Number of Movies'].append(num_movies)

    # Convert results to DataFrame
    results_df = pd.DataFrame(results)

    # Plot the results
    plt.figure(figsize=(10, 6))
    plt.bar(results_df['Percentage of Users Viewing'], results_df['Number of Movies'], width=step, align='center')
    plt.xlabel('Percentage of Total Users Viewing')
    plt.ylabel('Number of Movies')
    plt.title('Number of Movies Viewed by Different Percentages of Users')
    # plt.xticks(rotation=30)
    plt.show()

    return results_df

def filter_movies_by_threshold(df, threshold_percentage):
    """
    Filter the DataFrame based on a view count threshold.

    Parameters:
    - df (pd.DataFrame): The input DataFrame with columns 'userId' and 'movieId'.
    - threshold_percentage (float): The threshold percentage of the maximum view count.

    Returns:
    - pd.DataFrame: A filtered DataFrame where movies meet the view count threshold.
    """
    # Calculate the total number of unique users
    num_users = df['userId'].nunique()

    # Calculate the view count for each movie
    movie_view_counts = df['movieId'].value_counts()

    # Calculate the maximum view count (the most viewed movie)
    max_views = movie_view_counts.max()

    # Calculate the threshold view count
    threshold = max_views * (threshold_percentage / 100)

    # Filter the movies based on the threshold
    filtered_movies = movie_view_counts[movie_view_counts >= threshold].index
    filtered_df = df[df['movieId'].isin(filtered_movies)]

    return filtered_df

def load_movielens_data(file_path, num_users, num_movies):
    """
    Load the MovieLens dataset and transform it into a DataFrame.

    Parameters:
    - file_path (str): The path to the MovieLens data file.
    - num_users (int): The total number of users.
    - num_movies (int): The total number of movies.

    Returns:
    - pd.DataFrame: A DataFrame with users as rows, movies as columns, and ratings as values.
    """
    # Read the file into a DataFrame
    df = pd.read_csv(file_path, sep='\t', header=None, names=['user_id', 'movie_id', 'rating', 'timestamp'])

    # Pivot the DataFrame to get users as rows and movies as columns
    rating_matrix = df.pivot(index='user_id', columns='movie_id', values='rating')

    # Reindex to ensure the DataFrame has all users and movies
    rating_matrix = rating_matrix.reindex(index=range(1, num_users + 1), columns=range(1, num_movies + 1))

    # Fill missing values with 0 (or another value if you prefer)
    rating_matrix.fillna(0, inplace=True)

    return rating_matrix

def load_movielens_json(file_path):
    """
    Load the MovieLens dataset from a JSON file and transform it into a DataFrame.

    Parameters:
    - file_path (str): The path to the MovieLens JSON data file.

    Returns:
    - pd.DataFrame: A DataFrame with users as rows, movies as columns, and ratings as values.
    """
    # Read the JSON file line by line
    data = []
    with open(file_path, 'r') as file:
        for line in file:
            try:
                data.append(json.loads(line.strip()))
            except json.JSONDecodeError as e:
                print(f"Error decoding JSON: {e}")
                continue

    # Convert the list of dictionaries to a DataFrame
    df = pd.DataFrame(data)

    # Drop duplicate ratings based on user_id and item_id
    df = df.drop_duplicates(subset=['user_id', 'item_id'])

    # Create a pivot table without reindexing
    rating_matrix = df.pivot(index='user_id', columns='item_id', values='rating')

    # Fill missing values with 0 (or another value if you prefer)
    rating_matrix.fillna(0, inplace=True)

    return rating_matrix

def load_movielens_csv_to_matrix(file_path, num_users, num_movies):
    """
    Load the MovieLens dataset from a CSV file and transform it into a rating matrix DataFrame.

    Parameters:
    - file_path (str): The path to the MovieLens CSV data file.
    - num_users (int): The total number of users.
    - num_movies (int): The total number of movies.

    Returns:
    - pd.DataFrame: A DataFrame with user IDs as rows, movie IDs as columns, and ratings as values.
    """
    # Read the CSV file into a DataFrame
    df = pd.read_csv(file_path)

    # Drop the timestamp column as it is not needed
    df = df.drop(columns=['timestamp'])

    # Pivot the DataFrame to get users as rows and movies as columns
    rating_matrix = df.pivot(index='userId', columns='movieId', values='rating')

    # Reindex to ensure the DataFrame has all users and movies
    rating_matrix = rating_matrix.reindex(index=range(1, num_users + 1), columns=range(1, num_movies + 1))

    # Fill missing values with 0 (or another value if you prefer)
    rating_matrix.fillna(0, inplace=True)

    return rating_matrix

def df_to_matrix(df):
    """
    Convert the interaction DataFrame into a user-movie matrix with ratings as entries.

    Parameters:
    - df (pd.DataFrame): The input DataFrame with columns 'userId', 'movieId', and 'rating'.

    Returns:
    - pd.DataFrame: A user-movie matrix with ratings as entries.
    """
    # Create the pivot table to get users as rows and movies as columns
    user_movie_matrix = df.pivot(index='userId', columns='movieId', values='rating')

    # Fill missing values with 0 (or another value if you prefer)
    user_movie_matrix.fillna(0, inplace=True)

    return user_movie_matrix

def generate_tfidf_matrix(interactions_df):
    # Aggregate service class IDs per patient
    interactions_df['service_class_id'] = interactions_df['service_class_id'].astype(str)
    patient_services = interactions_df.groupby('patient_id')['service_class_id'].agg(' '.join).reset_index()

    # Count Vectorization
    cv = CountVectorizer(token_pattern='\\b\\w+\\b')  # Modify token pattern to include single characters
    cv_matrix = cv.fit_transform(patient_services['service_class_id'])

    # TF-IDF Transformation
    tfidf_transformer = TfidfTransformer()
    tfidf_matrix = tfidf_transformer.fit_transform(cv_matrix)

    # Convert to DataFrame with sorted columns
    tfidf_df = pd.DataFrame(tfidf_matrix.toarray(), index=patient_services['patient_id'], columns=cv.get_feature_names_out())
    tfidf_df.columns = tfidf_df.columns.astype(int)  # Convert column names to integers
    tfidf_df = tfidf_df.reindex(sorted(tfidf_df.columns), axis=1)  # Sort columns numerically

    return tfidf_df

def generate_tfidf_matrix_from_sequences(df):
    # Count Vectorization
    cv = CountVectorizer(token_pattern='\\b\\w+\\b')  # Modify token pattern to include single characters
    cv_matrix = cv.fit_transform(df['classes'])

    # TF-IDF Transformation
    tfidf_transformer = TfidfTransformer()
    tfidf_matrix = tfidf_transformer.fit_transform(cv_matrix)

    # Convert to DataFrame with sorted columns
    tfidf_df = pd.DataFrame(tfidf_matrix.toarray(), index=df['patient_id'], columns=cv.get_feature_names_out())
    tfidf_df.columns = tfidf_df.columns.astype(int)  # Convert column names to integers
    tfidf_df = tfidf_df.reindex(sorted(tfidf_df.columns), axis=1)  # Sort columns numerically

    return tfidf_df

def zero_one_transformation_numpy(tfidf_matrix):
    binary_matrix = (tfidf_matrix > 0).astype(int)
    return binary_matrix

def zero_one_transformation_torch(tfidf_matrix):
    tfidf_tensor = torch.tensor(tfidf_matrix)
    binary_tensor = (tfidf_tensor > 0).int().to(torch.float32)
    return binary_tensor

def normalize_rows_numpy(tfidf_matrix):
    norms = np.linalg.norm(tfidf_matrix, axis=1, keepdims=True)
    normalized_matrix = tfidf_matrix / norms
    return normalized_matrix

def normalize_rows_torch(tfidf_matrix):
    tfidf_tensor = torch.tensor(tfidf_matrix, dtype=torch.float32)
    norms = torch.norm(tfidf_tensor, p=2, dim=1, keepdim=True)
    normalized_tensor = tfidf_tensor / norms
    return normalized_tensor