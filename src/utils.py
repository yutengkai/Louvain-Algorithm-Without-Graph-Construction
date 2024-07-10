import pandas as pd
import numpy as np
import time
import torch
from tqdm import tqdm

from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
from sklearn.metrics.pairwise import cosine_similarity

from sklearn.metrics import adjusted_rand_score
import seaborn as sns
import matplotlib.pyplot as plt

def generate_patient_communities_df(patient_ids, community_assignments):
    """
    Generate a DataFrame with patient IDs and their corresponding community assignments.

    Parameters:
    - patient_ids (pd.Series): A series of patient IDs.
    - community_assignments (torch.Tensor): A tensor containing the community assignments for each patient.

    Returns:
    - pd.DataFrame: A DataFrame with patient IDs and their community assignments.
    """
    # Ensure community assignments is on CPU and convert to numpy array
    community_assignments_np = community_assignments.cpu().numpy()

    # Create the DataFrame
    patient_communities_df = pd.DataFrame({
        'patient_id': patient_ids,
        'community_assignment': community_assignments_np
    })

    return patient_communities_df

def compare_communities(df1, df2):
    """
    Compare two sets of community assignments using Adjusted Rand Index.

    Parameters:
    - df1 (pd.DataFrame): The first DataFrame with community assignments.
    - df2 (pd.DataFrame): The second DataFrame with community assignments.

    Returns:
    - float: The Adjusted Rand Index score.
    """
    # Ensure the patient IDs match
    assert df1['patient_id'].equals(df2['patient_id']), "Patient IDs do not match between the two DataFrames."

    # Calculate the Adjusted Rand Index
    ari_score = adjusted_rand_score(df1['community_assignment'], df2['community_assignment'])

    return ari_score

def plot_community_overlap(df1, df2):
    """
    Plot the overlap between two sets of community assignments.

    Parameters:
    - df1 (pd.DataFrame): The first DataFrame with community assignments.
    - df2 (pd.DataFrame): The second DataFrame with community assignments.
    """
    # Create a contingency table
    contingency_table = pd.crosstab(df1['community_assignment'], df2['community_assignment'])

    # Plot the heatmap
    plt.figure(figsize=(10, 8))
    sns.heatmap(contingency_table, annot=True, fmt="d", cmap="viridis")
    plt.xlabel('First Random Seed')
    plt.ylabel('Second Random Seed')
    plt.title('Community Assignment Overlap')
    plt.show()