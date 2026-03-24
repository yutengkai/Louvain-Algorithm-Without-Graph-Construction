"""
Basic tests for VLouvain algorithm.

These tests verify that the core functions work correctly on small synthetic data.
"""

import pytest
import torch
import sys
import os

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from src.main_algorithm import (
    node_degrees_torch,
    calculate_m_torch,
    louvain_partition_gpu,
    get_final_communities,
    modularity_all_partitions,
)


def create_test_matrix(n=100, d=32, device='cpu'):
    """Create a normalized test matrix for VLouvain."""
    torch.manual_seed(42)
    V = torch.randn(n, d, device=device)

    # L2 normalize
    norms = torch.linalg.norm(V, dim=1, keepdim=True)
    V_norm = V / norms

    # Augment with 1 and divide by sqrt(2)
    V_aug = torch.cat((V_norm, torch.ones(n, 1, device=device)), dim=1)
    V_aug = V_aug / (2 ** 0.5)

    return V_aug


class TestDegreeCalculation:
    """Tests for degree calculation functions."""

    def test_degrees_positive(self):
        """Degrees should be non-negative for normalized cosine similarity."""
        V = create_test_matrix(n=50, d=16)
        degrees = node_degrees_torch(V)
        assert torch.all(degrees >= 0), "Degrees should be non-negative"

    def test_degrees_shape(self):
        """Degree vector should have length n."""
        n = 50
        V = create_test_matrix(n=n, d=16)
        degrees = node_degrees_torch(V)
        assert degrees.shape == (n,), f"Expected shape ({n},), got {degrees.shape}"

    def test_m_positive(self):
        """Total edge weight m should be positive."""
        V = create_test_matrix(n=50, d=16)
        degrees = node_degrees_torch(V)
        m = calculate_m_torch(degrees)
        assert m > 0, "Total edge weight should be positive"


class TestModularity:
    """Tests for modularity calculation."""

    def test_modularity_range(self):
        """Modularity should be in [-0.5, 1] range."""
        V = create_test_matrix(n=50, d=16)
        partitions = louvain_partition_gpu(V, gamma=1.0, threshold=1e-7, seed=42)
        final = get_final_communities(partitions)
        Q = modularity_all_partitions(V, final)
        assert -0.5 <= Q <= 1.0, f"Modularity {Q} out of expected range"


class TestLouvainPartition:
    """Tests for the main Louvain algorithm."""

    def test_partition_output_type(self):
        """louvain_partition_gpu should return a list of tensors."""
        V = create_test_matrix(n=50, d=16)
        partitions = louvain_partition_gpu(V, gamma=1.0, threshold=1e-7, seed=42)
        assert isinstance(partitions, list), "Should return a list"
        assert all(isinstance(p, torch.Tensor) for p in partitions), "Each partition should be a tensor"

    def test_community_assignments_valid(self):
        """Community assignments should be valid indices."""
        n = 50
        V = create_test_matrix(n=n, d=16)
        partitions = louvain_partition_gpu(V, gamma=1.0, threshold=1e-7, seed=42)
        final = get_final_communities(partitions)

        assert final.shape == (n,), f"Expected shape ({n},), got {final.shape}"
        assert torch.all(final >= 0), "Community indices should be non-negative"

    def test_reproducibility(self):
        """Same seed should produce same results."""
        V = create_test_matrix(n=50, d=16)

        partitions1 = louvain_partition_gpu(V, gamma=1.0, threshold=1e-7, seed=42)
        final1 = get_final_communities(partitions1)

        partitions2 = louvain_partition_gpu(V, gamma=1.0, threshold=1e-7, seed=42)
        final2 = get_final_communities(partitions2)

        assert torch.equal(final1, final2), "Same seed should produce same partitions"

    def test_different_seeds_may_differ(self):
        """Different seeds may produce different results."""
        V = create_test_matrix(n=50, d=16)

        partitions1 = louvain_partition_gpu(V, gamma=1.0, threshold=1e-7, seed=42)
        final1 = get_final_communities(partitions1)

        partitions2 = louvain_partition_gpu(V, gamma=1.0, threshold=1e-7, seed=123)
        final2 = get_final_communities(partitions2)

        # They might be equal by chance, but at least they should both be valid
        assert final1.shape == final2.shape, "Both should have same shape"


class TestGetFinalCommunities:
    """Tests for the get_final_communities function."""

    def test_final_communities_shape(self):
        """Final communities should have same length as input nodes."""
        n = 50
        V = create_test_matrix(n=n, d=16)
        partitions = louvain_partition_gpu(V, gamma=1.0, threshold=1e-7, seed=42)
        final = get_final_communities(partitions)
        assert final.shape == (n,), f"Expected shape ({n},), got {final.shape}"

    def test_communities_contiguous(self):
        """Community indices should start from 0."""
        V = create_test_matrix(n=50, d=16)
        partitions = louvain_partition_gpu(V, gamma=1.0, threshold=1e-7, seed=42)
        final = get_final_communities(partitions)
        assert final.min() == 0, "Community indices should start from 0"


class TestResolutionParameter:
    """Tests for the gamma (resolution) parameter."""

    def test_higher_gamma_more_communities(self):
        """Higher gamma should generally produce more communities."""
        V = create_test_matrix(n=100, d=32)

        partitions_low = louvain_partition_gpu(V, gamma=0.5, threshold=1e-7, seed=42)
        final_low = get_final_communities(partitions_low)
        n_communities_low = final_low.max().item() + 1

        partitions_high = louvain_partition_gpu(V, gamma=2.0, threshold=1e-7, seed=42)
        final_high = get_final_communities(partitions_high)
        n_communities_high = final_high.max().item() + 1

        # Higher gamma should produce >= communities (not strictly > due to randomness)
        assert n_communities_high >= n_communities_low, \
            f"Higher gamma ({n_communities_high}) should produce >= communities than lower gamma ({n_communities_low})"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
