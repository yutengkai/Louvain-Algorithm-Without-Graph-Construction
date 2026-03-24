# VLouvain

**Vector-Based Louvain for Massive Dense Graphs**

[![DOI](https://img.shields.io/badge/DOI-10.48786%2Fedbt.2026.43-blue)](https://doi.org/10.48786/edbt.2026.43)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-ee4c2c.svg)](https://pytorch.org/)

VLouvain runs community detection **directly on embedding matrices** — no graph construction needed.

Traditional Louvain requires an explicit adjacency matrix, which becomes infeasible for dense similarity graphs (O(n²) memory). VLouvain keeps only the feature matrix (O(n·d) memory) and computes edge weights on-the-fly via inner products.

**Result:** VLouvain handles **1.5M+ node** graphs where cuGraph, iGraph, and NetworKit all crash.

---

## Quick Start

```python
import torch
from src.main_algorithm import louvain_partition_gpu, get_final_communities, modularity_all_partitions

# 1. Your embedding matrix (n nodes × d dimensions)
V = torch.randn(50_000, 512, device="cuda")

# 2. Normalize: L2-normalize, append 1, divide by √2
norms = torch.linalg.norm(V, dim=1, keepdim=True)
V_norm = V / norms
V_norm = torch.cat((V_norm, torch.ones(V_norm.size(0), 1, device=V_norm.device)), dim=1)
V_norm /= 2 ** 0.5

# 3. Run VLouvain
partitions = louvain_partition_gpu(V_norm, gamma=1.0, threshold=1e-7, seed=42)

# 4. Get results
communities = get_final_communities(partitions)
Q = modularity_all_partitions(V_norm, communities)

print(f"Found {communities.max().item() + 1} communities")
print(f"Modularity: {Q:.4f}")
```

**That's it.** No adjacency matrix, no k-NN graph, no edge list.

---

## Installation

```bash
git clone https://github.com/yutengkai/VLouvain.git
cd VLouvain
pip install -r requirements.txt
```

**For GPU acceleration** (recommended):
```bash
pip install torch --index-url https://download.pytorch.org/whl/cu118
```

---

## Why VLouvain?

| Method | 1.57M nodes (Amazon) | Memory |
|--------|---------------------|--------|
| **VLouvain** | ✅ 5,400 sec | O(n·d) |
| cuGraph | ❌ OOM | O(n²) |
| iGraph | ❌ OOM | O(n²) |
| NetworKit | ❌ OOM | O(n²) |
| GVE-Louvain | ❌ OOM | O(n²) |

*From Table 2 in the paper. All methods use A100 GPU / 83.5 GB RAM.*

### The Key Insight

Instead of storing edge weights explicitly:
```
A[i,j] = cosine_similarity(v_i, v_j)  # O(n²) storage
```

VLouvain computes them implicitly:
```
A[i,j] = v_i · v_j  # computed on-the-fly, O(n·d) storage
```

The math is identical — only the bookkeeping changes.

---

## API Reference

### Main Functions

| Function | Description |
|----------|-------------|
| `louvain_partition_gpu(V, gamma, threshold, seed)` | Run hierarchical Louvain on normalized matrix V |
| `get_final_communities(partitions)` | Flatten hierarchy to final node→community mapping |
| `modularity_all_partitions(V, communities)` | Compute modularity score |

### Parameters

| Parameter | Default | Description |
|-----------|---------|-------------|
| `gamma` | 1.0 | Resolution parameter. Higher = more communities |
| `threshold` | 1e-7 | Stop when ΔQ < threshold |
| `seed` | 42 | Random seed for reproducibility |

### Input Format

Your input matrix must be **normalized**:

1. **L2-normalize** each row to unit length
2. **Append a column of 1s**
3. **Divide by √2**

This transforms cosine similarity from [-1, 1] to [0, 1]:
```
A[i,j] = (1 + cos(θ)) / 2 ∈ [0, 1]
```

---

## Notebooks

Run experiments from the paper:

| Notebook | What it does | Colab |
|----------|--------------|-------|
| `Paper_Notebook_Homogeneous.ipynb` | VLouvain vs cuGraph on Flickr/Yelp/Amazon | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/yutengkai/VLouvain/blob/main/notebooks/Paper_Notebook_Homogeneous.ipynb) |
| `Paper_Notebook_Homogeneous_IGraph.ipynb` | VLouvain vs iGraph | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/yutengkai/VLouvain/blob/main/notebooks/Paper_Notebook_Homogeneous_IGraph.ipynb) |
| `Paper_Notebook_Homogeneous_other_libraries.ipynb` | VLouvain vs NetworKit, GVE | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/yutengkai/VLouvain/blob/main/notebooks/Paper_Notebook_Homogeneous_other_libraries.ipynb) |

---

## Datasets

The paper uses these PyTorch Geometric datasets:

| Dataset | Nodes | Dimensions | Load with |
|---------|-------|------------|-----------|
| Flickr | 89,250 | 500 | `Flickr(root="data/Flickr")` |
| Yelp | 716,847 | 300 | `Yelp(root="data/Yelp")` |
| Taobao | 936,946 | 66 | `Taobao(root="data/Taobao")` |
| Amazon Products | 1,569,960 | 200 | `AmazonProducts(root="data/Amazon")` |

```python
from torch_geometric.datasets import Flickr
dataset = Flickr(root="data/Flickr")
X = dataset.data.x  # Node features
```

---

## Application: GraphRAG

VLouvain powers [GraphRAG-V](https://doi.org/10.1007/978-3-031-78093-6_1), a faster alternative to Microsoft's GraphRAG:

| Metric | GraphRAG | GraphRAG-V |
|--------|----------|------------|
| Index build time | 3 hours | **5.3 minutes** |
| Retrieval recall | 37.9% | **48.8%** |

See the [GraphRAG-V paper](https://doi.org/10.1007/978-3-031-78093-6_1) for details.

---

## Citation

If you use VLouvain in your research, please cite:

```bibtex
@inproceedings{yu2026vlouvain,
  title     = {Efficient Vector-Based Louvain Algorithm for Massive Low-Rank Graphs},
  author    = {Yu, Tengkai and Srinivasan, Venkatesh and Thomo, Alex},
  booktitle = {Proceedings of the 29th International Conference on Extending Database Technology (EDBT)},
  year      = {2026},
  pages     = {537--543},
  doi       = {10.48786/edbt.2026.43},
  publisher = {OpenProceedings.org},
  address   = {Tampere, Finland}
}
```

---

## License

MIT License. See [LICENSE](LICENSE) for details.

---

## FAQ

**Q: Does VLouvain work on sparse graphs?**
A: Yes, but traditional Louvain is already efficient for sparse graphs. VLouvain shines when the graph is dense (similarity graphs, embeddings).

**Q: Can I use a different similarity function?**
A: The current implementation uses normalized cosine similarity. For other similarity functions, you'd need to modify the normalization step.

**Q: What resolution parameter (γ) should I use?**
A: Start with γ=1.0 (default). Higher values produce more, smaller communities.

---

## Contact

- **Tengkai Yu** — yutengkai@uvic.ca
- **Venkatesh Srinivasan** — vsrinivasan4@scu.edu
- **Alex Thomo** — thomo@uvic.ca
