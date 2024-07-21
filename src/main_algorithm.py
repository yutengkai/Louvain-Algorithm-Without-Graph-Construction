import pandas as pd
import numpy as np
import time
import torch
from tqdm import tqdm

from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
from sklearn.metrics.pairwise import cosine_similarity

def node_degrees_numpy(matrix):
    # Sum up the entire matrix into a single row vector
    row_sums = np.sum(matrix, axis=0)

    # Calculate the degrees for each node
    degrees = np.dot(matrix, row_sums) - np.sum(matrix * matrix, axis=1)

    return degrees

def node_degrees_torch(matrix):
    # Sum up the entire matrix into a single row vector
    row_sums = torch.sum(matrix, dim=0)

    # Calculate the degrees for each node
    degrees = torch.matmul(matrix, row_sums) - torch.sum(matrix * matrix, dim=1)

    return degrees

def calculate_m_numpy(degrees):
    return np.sum(degrees) / 2


def calculate_m_torch(degrees):
    return torch.sum(degrees) / 2

def initialize_communities_numpy(matrix, shuffle=True, seed=42):
    num_nodes = matrix.shape[0]

    # Initialize community assignments: each node in its own community
    community_assignments = np.arange(num_nodes)

    # Initialize visiting order
    np.random.seed(seed)
    if shuffle:
        visit_order = np.random.permutation(num_nodes)
    else:
        visit_order = np.arange(num_nodes)

    return community_assignments, visit_order

def initialize_communities_torch(matrix, shuffle=True, seed=42):
    num_nodes = matrix.shape[0]

    # Initialize community assignments: each node in its own community
    community_assignments = torch.arange(num_nodes)

    # Initialize visiting order
    if shuffle:
        torch.manual_seed(seed)
        visit_order = torch.randperm(num_nodes)
    else:
        visit_order = torch.arange(num_nodes)

    return community_assignments, visit_order


def q_assign_cpu(matrix, degrees, m, gamma, community_assignments, vertex_index, community_index):
    X_i = matrix[vertex_index]
    k_i = degrees[vertex_index]

    if community_index != community_assignments[vertex_index]:
        # Calculate for the case when i is not in C
        sum_X_j = np.sum(matrix[community_assignments==community_index], axis=0)
        inner_product = np.inner(X_i, sum_X_j)
        sum_k_j = np.sum(degrees[community_assignments[community_index]])
        q_contrib = inner_product - (k_i * sum_k_j) / (2 * m)
    else:
        # Calculate for the case when i is in C
        sum_X_j = np.sum(matrix[community_assignments==community_index], axis=0)
        inner_product = np.inner(X_i, sum_X_j)
        sum_k_j = np.sum(degrees[community_assignments[community_index]])
        q_contrib = inner_product - (k_i * sum_k_j) / (2 * m) - np.inner(X_i, X_i) + (k_i ** 2) / (2 * m)
    return q_contrib / (2 * m)

def q_assign_gpu(matrix, degrees, m, gamma, community_assignments, vertex_index, community_index):
    X_i = matrix[vertex_index]
    k_i = degrees[vertex_index]

    if community_index != community_assignments[vertex_index]:
        # Calculate for the case when i is not in C
        sum_X_j = torch.sum(matrix[community_assignments == community_index], dim=0)
        inner_product = torch.inner(X_i, sum_X_j)
        sum_k_j = torch.sum(degrees[community_assignments == community_index])
        q_contrib = inner_product - (k_i * sum_k_j) / (2 * m)
    else:
        # Calculate for the case when i is in C
        sum_X_j = torch.sum(matrix[community_assignments == community_index], dim=0)
        inner_product = torch.inner(X_i, sum_X_j)
        sum_k_j = torch.sum(degrees[community_assignments == community_index])
        q_contrib = inner_product - (k_i * sum_k_j) / (2 * m) - torch.inner(X_i, X_i) + (k_i ** 2) / (2 * m)

    return q_contrib / (2 * m)

def optimized_community_cpu(matrix, community_matrix, degrees, community_degrees, m, gamma, community_assignments, vertex_index):
    X_i = matrix[vertex_index]
    k_i = degrees[vertex_index]

    # Compute the inner product of X_i with the community matrix
    inner_products = np.inner(X_i, community_matrix)

    # Compute degree contributions
    degree_contributions = k_i * community_degrees / (2 * m)

    # Compute Q contributions for all communities
    Q_contributions = inner_products - degree_contributions

    # Handle the case where i is in its current community
    current_community = community_assignments[vertex_index]
    same_community_mask = (community_assignments == current_community)
    Q_contributions[same_community_mask] -= (np.inner(X_i, X_i) - (k_i ** 2) / (2 * m))

    # Find the optimal community and the corresponding delta Q
    optimal_community = np.argmax(Q_contributions)
    delta_Q = Q_contributions[optimal_community] - Q_contributions[current_community]

    if optimal_community != current_community:
        # Update the community assignments and degrees
        community_assignments[vertex_index] = optimal_community
        community_matrix[same_community_mask] -= X_i
        community_matrix[community_assignments == optimal_community] += X_i
        community_degrees[same_community_mask] -= k_i
        community_degrees[community_assignments == optimal_community] += k_i

    return optimal_community, delta_Q / (2 * m)


def optimized_community_gpu(matrix, community_matrix, degrees, community_degrees, m, gamma, community_assignments, vertex_index):
    X_i = matrix[vertex_index]
    k_i = degrees[vertex_index]

    # Compute the inner product of X_i with the community matrix
    inner_products = torch.matmul(X_i, community_matrix.t())

    # Compute degree contributions
    degree_contributions = gamma * k_i * community_degrees / (2 * m)

    # Compute Q contributions for all communities
    Q_contributions = inner_products - degree_contributions

    # Handle the case where i is in its current community
    current_community = community_assignments[vertex_index]
    same_community_mask = (community_assignments == current_community)
    Q_contributions[same_community_mask] -= (torch.dot(X_i, X_i) - gamma * (k_i ** 2) / (2 * m))

    # Find the optimal community and the corresponding delta Q
    optimal_community_index = torch.argmax(Q_contributions)
    optimal_community = community_assignments[optimal_community_index]
    delta_Q = Q_contributions[optimal_community_index] - Q_contributions[vertex_index]

    if optimal_community != current_community:
        # Update the community assignments
        community_assignments[vertex_index] = optimal_community

        # Update the community matrix and degrees for current and optimal communities
        community_matrix[same_community_mask] -= X_i
        community_matrix[community_assignments == optimal_community] += X_i
        community_matrix[vertex_index] = community_matrix[optimal_community_index]

        community_degrees[same_community_mask] -= k_i
        community_degrees[community_assignments == optimal_community] += k_i
        community_degrees[vertex_index] = community_degrees[optimal_community_index]

    return optimal_community.item(), delta_Q.item() / (2 * m)


def one_level_cpu(matrix, degrees, m, gamma):
    # Initialize communities and visiting order
    community_assignments, visit_order = initialize_communities_numpy(matrix, shuffle=True, seed=42)
    community_matrix = np.copy(matrix)
    community_degrees = np.copy(degrees)
    total_Q = 0.0

    # Iterate through the visit order with a progress bar
    for vertex_index in tqdm(visit_order, desc="Processing nodes"):
        # Get the optimal community and delta Q
        optimal_community, delta_Q = optimized_community_cpu(matrix, community_matrix, degrees, community_degrees, m, gamma, community_assignments, vertex_index)

        # Update community assignments and total modularity
        community_assignments[vertex_index] = optimal_community
        total_Q += delta_Q

    # Get the number of unique communities
    unique_communities = len(np.unique(community_assignments))

    return unique_communities, total_Q, community_assignments

def one_level_gpu(matrix, degrees, m, gamma, seed=42):
    # Initialize communities and visiting order
    community_assignments, visit_order = initialize_communities_torch(matrix, shuffle=True, seed=seed)
    community_matrix = matrix.clone()
    community_degrees = degrees.clone()
    total_Q = torch.tensor(0.0, device=matrix.device)

    # Move community_assignments and visit_order to the same device as matrix
    community_assignments = community_assignments.to(matrix.device)
    visit_order = visit_order.to(matrix.device)

    # Iterate through the visit order with a progress bar
    for vertex_index in tqdm(visit_order, desc="Processing nodes"):
        # Get the optimal community and delta Q
        optimal_community, delta_Q = optimized_community_gpu(matrix, community_matrix, degrees, community_degrees, m, gamma, community_assignments, vertex_index)

        # Update community assignments and total modularity
        community_assignments[vertex_index] = optimal_community
        total_Q += delta_Q

    # Retrieve unique communities and their corresponding rows in the community matrix
    unique_communities, indices = torch.unique(community_assignments, return_inverse=True)
    unique_community_indices = torch.tensor([torch.where(community_assignments == u)[0][0] for u in unique_communities], device=matrix.device)
    unique_community_matrix = community_matrix[unique_community_indices]
    unique_community_degrees = community_degrees[unique_community_indices]

    new_community_assignments = indices

    # Create a dictionary for the output
    result = {
    "num_communities": len(unique_communities),
    "total_Q": total_Q.item(),
    "community_assignments": new_community_assignments,
    "unique_community_matrix": unique_community_matrix,
    "unique_community_degrees": unique_community_degrees
}

    return result

def louvain_partition_gpu(matrix, gamma=1.0, threshold=1e-7, max_level=-1, seed=42, print_layer_result=False):
    # Initialize degrees and modularity
    degrees = node_degrees_torch(matrix)
    m = calculate_m_torch(degrees)
    total_modularity = 0.0

    # Store results
    partitions = []

    # Start with the original matrix
    current_matrix = matrix
    current_degrees = degrees
    current_gamma = gamma
    current_m = m

    while True:
        # Call one_level_gpu function
        result = one_level_gpu(current_matrix, current_degrees, current_m, current_gamma, seed)

        # Extract the results
        num_communities = result["num_communities"]
        delta_Q = result["total_Q"]
        community_assignments = result["community_assignments"]
        unique_community_matrix = result["unique_community_matrix"]
        unique_community_degrees = result["unique_community_degrees"]

        # Print layer results if needed
        if print_layer_result:
            print(f"Number of communities: {num_communities}, Delta Q: {delta_Q}")
            print(torch.unique(unique_community_matrix, dim=0).shape)
        # Update total modularity
        total_modularity += delta_Q

        # Append the current community assignments
        partitions.append(community_assignments)

        # Check for termination condition
        if delta_Q <= threshold or (max_level > 0 and len(partitions) >= max_level):
            break

        # Update the current matrix and degrees for the next level
        current_matrix = unique_community_matrix
        # current_degrees = node_degrees_torch(current_matrix)
        current_degrees = unique_community_degrees

    return partitions

def louvain_communities_gpu(matrix, gamma=1.0, threshold=1e-7, max_level=-1, seed=42, print_layer_result=False):
    # Call the louvain_partition_gpu function and get the partitions
    partitions = louvain_partition_gpu(matrix, gamma, threshold, max_level, seed, print_layer_result)

    # Return the final partition (last element in the list)
    return partitions[-1]

def get_final_communities(partitions):
    """
    Given a list of partitions from the Louvain algorithm, map the original nodes to their final communities.

    Parameters:
    - partitions (list of torch.Tensor): List of partition tensors, where each tensor represents
      the community assignments at a given level of the algorithm.

    Returns:
    - torch.Tensor: A tensor of community assignments for the original nodes.
    """
    # Start with the last partition (the most reduced one)
    final_partition = partitions[-1]
    # Iterate backwards through the partitions to map to original nodes
    for partition in reversed(partitions[:-1]):
        final_partition = final_partition[partition]

    return final_partition

def modularity_single_community(matrix, degrees, m):
    """
    Calculate the modularity for a single community.

    Parameters:
    - matrix (torch.Tensor): The input adjacency matrix.
    - degrees (torch.Tensor): The degree vector.
    - m (float): The total weight of the graph.

    Returns:
    - float: The modularity of the single community.
    """
    # Calculate community row (sum of matrix rows corresponding to the community)
    community_row = torch.sum(matrix, dim=0)

    # Calculate community degree
    community_degree = torch.sum(degrees)

    # Calculate the modularity
    inner_product_community = torch.dot(community_row, community_row)
    degree_contribution = (community_degree ** 2) / (2 * m)

    modularity = inner_product_community - degree_contribution

    # Calculate the self-interaction part
    self_interaction = torch.dot(matrix.flatten(), matrix.flatten().t())
    self_interaction -= torch.dot(degrees, degrees) / (2 * m)

    # Subtract self-interaction and normalize by 2m
    modularity -= self_interaction
    modularity /= 2 * m

    return modularity.item()

def modularity_all_partitions(matrix, partitions):
    """
    Calculate the modularity for all partitions.

    Parameters:
    - matrix (torch.Tensor): The input adjacency matrix.
    - partitions (torch.Tensor): The partition vector indicating community assignments.
    - degrees (torch.Tensor): The degree vector.
    - m (float): The total weight of the graph.

    Returns:
    - float: The total modularity for all partitions.
    """
    degrees = node_degrees_torch(matrix)
    m = calculate_m_torch(degrees)

    unique_communities = torch.unique(partitions)
    total_modularity = 0.0

    for community in unique_communities:
        community_mask = (partitions == community)
        community_matrix = matrix[community_mask]
        community_degrees = degrees[community_mask]
        # print(modularity_single_community(community_matrix, community_degrees, m))
        total_modularity += modularity_single_community(community_matrix, community_degrees, m)

    return total_modularity

def node_degrees_pos_neg(matrix):
    # Compute row sums and absolute row sums
    row_sums = torch.sum(matrix, dim=0)
    abs_row_sums = torch.sum(torch.abs(matrix), dim=0)

    # Compute degrees (positive - negative) and abs_degrees (positive + negative)
    degrees = torch.matmul(matrix, row_sums) - torch.sum(matrix * matrix, dim=1)
    abs_degrees = torch.matmul(torch.abs(matrix), abs_row_sums) - torch.sum(matrix * matrix, dim=1)

    # Compute positive and negative degrees
    degree_pos = (degrees + abs_degrees) / 2
    degree_neg = (abs_degrees - degrees) / 2

    return degree_pos, degree_neg

def calculate_m_pos_neg(degree_pos, degree_neg):
    # Calculate total positive degrees and total negative degrees
    total_pos_degrees = torch.sum(degree_pos)
    total_neg_degrees = torch.sum(degree_neg)

    # Calculate m_pos and m_neg
    m_pos = total_pos_degrees / 2
    m_neg = total_neg_degrees / 2

    return m_pos, m_neg

def initialize_communities(matrix, shuffle=True, seed=42):
    num_nodes = matrix.shape[0]

    # Initialize community assignments: each node in its own community
    community_assignments = torch.arange(num_nodes)

    # Initialize visiting order
    if shuffle:
        torch.manual_seed(seed)
        visit_order = torch.randperm(num_nodes)
    else:
        visit_order = torch.arange(num_nodes)

    return community_assignments, visit_order

def optimized_community_pos_neg(matrix, community_matrix, absolute_community_matrix, degree_pos, degree_neg, community_degrees_pos, community_degrees_neg, m_pos, m_neg, gamma, community_assignments, vertex_index):
    X_i = matrix[vertex_index]
    abs_X_i = torch.abs(X_i)
    k_i_pos = degree_pos[vertex_index]
    k_i_neg = degree_neg[vertex_index]

    # Compute the inner product of X_i with the community matrix and absolute community matrix
    inner_products_diff = torch.matmul(X_i, community_matrix.t())
    inner_products_sum = torch.matmul(abs_X_i, absolute_community_matrix.t())

    # Derive inner_products_pos and inner_products_neg
    inner_products_pos = (inner_products_diff + inner_products_sum) / 2
    inner_products_neg = (inner_products_sum - inner_products_diff) / 2

    # Compute degree contributions
    degree_contributions_pos = gamma * k_i_pos * community_degrees_pos / (2 * m_pos)

    # Compute Q contributions for positive graph
    Q_contributions_pos = inner_products_pos - degree_contributions_pos

    # Handle the case where i is in its current community
    current_community = community_assignments[vertex_index]
    same_community_mask = (community_assignments == current_community)
    Q_contributions_pos[same_community_mask] -= (torch.dot(X_i, X_i) - gamma * (k_i_pos ** 2) / (2 * m_pos))

    # Divide Q contributions for positive graph by 2 * m_pos
    Q_contributions_pos /= (2 * m_pos)

    if m_neg > 0:
        # Compute degree contributions for negative graph
        degree_contributions_neg = gamma * k_i_neg * community_degrees_neg / (2 * m_neg)

        # Compute Q contributions for negative graph
        Q_contributions_neg = inner_products_neg - degree_contributions_neg

        # Handle the case where i is in its current community for negative graph
        Q_contributions_neg[same_community_mask] -= gamma * (k_i_neg ** 2) / (2 * m_neg)

        # Divide Q contributions for negative graph by 2 * m_neg
        Q_contributions_neg /= (2 * m_neg)

        # Combine Q contributions
        Q_contributions = Q_contributions_pos - Q_contributions_neg
    else:
        Q_contributions = Q_contributions_pos

    # Find the optimal community and the corresponding delta Q
    optimal_community_index = torch.argmax(Q_contributions)
    optimal_community = community_assignments[optimal_community_index]
    delta_Q = Q_contributions[optimal_community_index] - Q_contributions[vertex_index]

    if optimal_community != current_community:
        # Update the community assignments
        community_assignments[vertex_index] = optimal_community

        # Update the community matrix and degrees for current and optimal communities
        community_matrix[same_community_mask] -= X_i
        community_matrix[community_assignments == optimal_community] += X_i
        community_matrix[vertex_index] = community_matrix[optimal_community_index]

        absolute_community_matrix[same_community_mask] -= abs_X_i
        absolute_community_matrix[community_assignments == optimal_community] += abs_X_i
        absolute_community_matrix[vertex_index] = absolute_community_matrix[optimal_community_index]

        community_degrees_pos[same_community_mask] -= k_i_pos
        community_degrees_pos[community_assignments == optimal_community] += k_i_pos
        community_degrees_pos[vertex_index] = community_degrees_pos[optimal_community_index]

        community_degrees_neg[same_community_mask] -= k_i_neg
        community_degrees_neg[community_assignments == optimal_community] += k_i_neg
        community_degrees_neg[vertex_index] = community_degrees_neg[optimal_community_index]

    return optimal_community.item(), delta_Q.item()

def one_level_pos_neg(matrix, degree_pos, degree_neg, m_pos, m_neg, gamma, seed=42):
    # Initialize communities and visiting order
    community_assignments, visit_order = initialize_communities(matrix, shuffle=True, seed=seed)
    community_matrix = matrix.clone()
    absolute_community_matrix = torch.abs(matrix.clone())
    community_degrees_pos = degree_pos.clone()
    community_degrees_neg = degree_neg.clone()
    total_Q = torch.tensor(0.0, device=matrix.device)

    # Move community_assignments and visit_order to the same device as matrix
    community_assignments = community_assignments.to(matrix.device)
    visit_order = visit_order.to(matrix.device)

    # Iterate through the visit order with a progress bar
    for vertex_index in tqdm(visit_order, desc="Processing nodes"):
        # Get the optimal community and delta Q
        optimal_community, delta_Q = optimized_community_pos_neg(
            matrix, community_matrix, absolute_community_matrix, degree_pos, degree_neg,
            community_degrees_pos, community_degrees_neg, m_pos, m_neg, gamma,
            community_assignments, vertex_index
        )

        # Update community assignments and total modularity
        community_assignments[vertex_index] = optimal_community
        total_Q += delta_Q

    # Retrieve unique communities and their corresponding rows in the community matrix
    unique_communities, indices = torch.unique(community_assignments, return_inverse=True)
    unique_community_indices = torch.tensor([torch.where(community_assignments == u)[0][0] for u in unique_communities], device=matrix.device)
    unique_community_matrix = community_matrix[unique_community_indices]
    unique_community_degrees_pos = community_degrees_pos[unique_community_indices]
    unique_community_degrees_neg = community_degrees_neg[unique_community_indices]

    new_community_assignments = indices

    # Create a dictionary for the output
    result = {
        "num_communities": len(unique_communities),
        "total_Q": total_Q.item(),
        "community_assignments": new_community_assignments,
        "unique_community_matrix": unique_community_matrix,
        "unique_community_degrees_pos": unique_community_degrees_pos,
        "unique_community_degrees_neg": unique_community_degrees_neg
    }

    return result

def louvain_partition_pos_neg(matrix, gamma=1.0, threshold=1e-7, max_level=-1, seed=42, print_layer_result=False):
    # Initialize degrees and modularity
    degree_pos, degree_neg = node_degrees_pos_neg(matrix)
    m_pos, m_neg = calculate_m_pos_neg(degree_pos, degree_neg)
    total_modularity = 0.0

    # Store results
    partitions = []

    # Start with the original matrix
    current_matrix = matrix
    current_degree_pos = degree_pos
    current_degree_neg = degree_neg
    current_m_pos = m_pos
    current_m_neg = m_neg
    current_gamma = gamma

    while True:
        # Call one_level_pos_neg function
        result = one_level_pos_neg(current_matrix, current_degree_pos, current_degree_neg, current_m_pos, current_m_neg, current_gamma, seed)

        # Extract the results
        num_communities = result["num_communities"]
        delta_Q = result["total_Q"]
        community_assignments = result["community_assignments"]
        unique_community_matrix = result["unique_community_matrix"]
        unique_community_degrees_pos = result["unique_community_degrees_pos"]
        unique_community_degrees_neg = result["unique_community_degrees_neg"]

        # Print layer results if needed
        if print_layer_result:
            print(f"Number of communities: {num_communities}, Delta Q: {delta_Q}")
            print(torch.unique(unique_community_matrix, dim=0).shape)

        # Update total modularity
        total_modularity += delta_Q

        # Append the current community assignments
        partitions.append(community_assignments)

        # Check for termination condition
        if delta_Q <= threshold or (max_level > 0 and len(partitions) >= max_level):
            break

        # Update the current matrix and degrees for the next level
        current_matrix = unique_community_matrix
        current_degree_pos = unique_community_degrees_pos
        current_degree_neg = unique_community_degrees_neg
        current_m_pos = m_pos
        current_m_neg = m_neg

    return partitions

def modularity_single_community_pos_neg(matrix, degree_pos, degree_neg, m_pos, m_neg):
    """
    Calculate the modularity for a single community considering both positive and negative weights.

    Parameters:
    - matrix (torch.Tensor): The input adjacency matrix.
    - degree_pos (torch.Tensor): The positive degree vector.
    - degree_neg (torch.Tensor): The negative degree vector.
    - m_pos (float): The total positive weight of the graph.
    - m_neg (float): The total negative weight of the graph.

    Returns:
    - float: The modularity of the single community.
    """
    # Create the absolute value matrix
    absolute_matrix = torch.abs(matrix)

    # Calculate community row (sum of matrix rows corresponding to the community)
    community_row = torch.sum(matrix, dim=0)
    absolute_community_row = torch.sum(absolute_matrix, dim=0)

    # Compute inner product community differences and sums
    inner_product_community_diff = torch.dot(community_row, community_row)
    inner_product_community_sum = torch.dot(absolute_community_row, absolute_community_row)

    # Decompose into positive and negative components
    inner_product_community_pos = (inner_product_community_diff + inner_product_community_sum) / 2
    inner_product_community_neg = (inner_product_community_sum - inner_product_community_diff) / 2

    # Calculate degree contributions
    degree_contribution_pos = (torch.sum(degree_pos) ** 2) / (2 * m_pos)

    # Calculate modularity for positive weights
    modularity_pos = inner_product_community_pos - degree_contribution_pos

    # Subtract self-interaction and normalize by 2m_pos
    self_interaction_pos = torch.dot(matrix.flatten(), matrix.flatten().t())
    self_interaction_pos -= torch.dot(degree_pos, degree_pos) / (2 * m_pos)
    modularity_pos -= self_interaction_pos
    modularity_pos /= 2 * m_pos

    if m_neg > 0:
        # Calculate degree contributions for negative weights
        degree_contribution_neg = (torch.sum(degree_neg) ** 2) / (2 * m_neg)

        # Calculate modularity for negative weights
        modularity_neg = inner_product_community_neg - degree_contribution_neg

        # Subtract self-interaction and normalize by 2m_neg
        self_interaction_neg = -torch.dot(degree_neg, degree_neg) / (2 * m_neg)
        modularity_neg -= self_interaction_neg
        modularity_neg /= 2 * m_neg

        # Combine positive and negative modularities
        modularity = modularity_pos - modularity_neg
    else:
        modularity = modularity_pos

    return modularity.item()


def modularity_all_partitions_pos_neg(matrix, partitions):
    """
    Calculate the modularity for all partitions considering both positive and negative weights.

    Parameters:
    - matrix (torch.Tensor): The input adjacency matrix.
    - partitions (torch.Tensor): The partition vector indicating community assignments.

    Returns:
    - float: The total modularity for all partitions.
    """
    degree_pos, degree_neg = node_degrees_pos_neg(matrix)
    m_pos, m_neg = calculate_m_pos_neg(degree_pos, degree_neg)

    unique_communities = torch.unique(partitions)
    total_modularity = 0.0

    for community in unique_communities:
        community_mask = (partitions == community)
        community_matrix = matrix[community_mask]
        community_degrees_pos = degree_pos[community_mask]
        community_degrees_neg = degree_neg[community_mask]

        total_modularity += modularity_single_community_pos_neg(
            community_matrix, community_degrees_pos, community_degrees_neg, m_pos, m_neg
        )

    return total_modularity