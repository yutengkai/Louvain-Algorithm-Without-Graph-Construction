# VLouvain – Graph-less Louvain for Massive Dense Graphs
*A vector-based community-detection algorithm that scales to millions of nodes by skipping explicit edge construction.*

## TL;DR

VLouvain runs the Louvain algorithm directly on node-embedding matrices.  
Instead of materialising an *n × n* similarity graph (O(n²) memory), it keeps only the feature matrix (*n × d*) and a few aggregated vectors (O(n d) memory).  
On dense graphs with >1 M nodes, VLouvain is the **only** open implementation we found that finishes on commodity hardware while traditional Louvain, Leiden, cuGraph and iGraph exhaust GPU/CPU memory. :contentReference[oaicite:1]{index=1}

---

## Repository structure

```

.
├── notebooks/                      # Jupyter demos used in the paper
│   ├── Paper\_Notebook\_Homogeneous.ipynb      # VLouvain vs cuGraph
│   ├── Paper\_Notebook\_Homogeneous\_Igraph.ipynb # VLouvain vs iGraph
│   └── Paper\_Notebook\_Homogeneous\_OtherLibs.ipynb # VLouvain vs other libs
├── src/
│   ├── main\_algorithm.py           # core VLouvain implementation
│   ├── utils.py
│   └── data/
│       ├── data\_downloading.py     # helper: fetch PYG datasets
│       └── data\_preprocessing.py   # helper: build graphs from vectors
├── tests/                          # unit tests (pytest)
├── requirements.txt
└── README.md   ← **you are here**

````

---

## Installation

```bash
git clone https://github.com/yutengkai/VLouvain.git
cd VLouvain
pip install -r requirements.txt        # PyTorch ≥2.1, NumPy, NetworkX, etc.
````

> **GPU:**
> If you want CUDA acceleration, install the matching PyTorch build first, e.g.
> `pip install torch==2.2.0+cu118 torchvision==0.17.0+cu118 -f https://download.pytorch.org/whl/torch_stable.html`

---

## Quick start (Python)

```python
from src.data.data_downloading import download_dataset
from src.data.data_preprocessing import build_graph
from src.main_algorithm import run_main_algorithm   # returns community labels

# 1 Download Flickr embeddings (~89 k nodes, 500-d)
raw = download_dataset("Flickr", data_dir="./data")

# 2 (Optional) preprocess – here we already have normalised vectors
graph = build_graph(raw)            # wraps vectors → graph object with .vectors

# 3 Run VLouvain
labels = run_main_algorithm(
    graph,
    resolution=1.0,
    device="cuda"        # set "cpu" if no GPU
)
print(f"Found {len(set(labels))} communities!")
```

---

## Notebook gallery

| Notebook                                     | What it shows                                       | Main comparators |
| -------------------------------------------- | --------------------------------------------------- | ---------------- |
| `Paper_Notebook_Homogeneous.ipynb`           | Flickr/Yelp/Amazon experiments; VLouvain vs cuGraph | cuGraph Louvain  |
| `Paper_Notebook_Homogeneous_IGraph.ipynb`    | Same datasets; VLouvain vs iGraph                   | iGraph Louvain   |
| `Paper_Notebook_Homogeneous_other_libraries.ipynb` | Benchmarks vs NetworKit & GVE-Louvain               | NetworKit, GVE   |

\* Measured on Google Colab Pro (A100 40 GB). See paper §6 for details.&#x20;

Each notebook is self-contained; launch with JupyterLab or click the Colab badge at the top of the file.

---

## Algorithm highlights (paper excerpts)

* **Memory efficiency** – keeps only feature matrix (*n × d*) plus four O(n) vectors, versus O(n²) edges in dense graphs.&#x20;
* **Mathematical equivalence** – identical ΔQ gain formula; only the bookkeeping differs.
* **Performance** – on the 1.57 M-node Amazon Products graph, VLouvain finishes in **\~5 400 s**, while cuGraph, iGraph, NetworKit and GVE all fail due to memory limits (Table 2, p. 6).&#x20;
* **GraphRAG application** – swapping GraphRAG’s LLM-built graph with VLouvain communities cuts index-build time from 3 h to 5 min and boosts retrieval recall by +11 % (p. 6–7).&#x20;

---


## FAQ

| Question                               | Answer                                                                                                       |
| -------------------------------------- | ------------------------------------------------------------------------------------------------------------ |
| **Import error** `my_louvain.src...`?  | Older notebooks used the legacy namespace. Replace with `from src.main_algorithm import run_main_algorithm`. |
| Does VLouvain work on *sparse* graphs? | Yes, but classic Louvain is already efficient there; VLouvain shines when the graph is dense or complete.    |
| Can I tune resolution γ?               | Pass `resolution=<float>` to `run_main_algorithm`; γ > 1 yields smaller communities.                         |

---

## Citation

Please cite both the paper and this repository:

```bibtex
@inproceedings{yu2025vlouvain,
  title     = {Efficient Vector-Based Louvain Algorithm for Massive Dense Graphs},
  author    = {Yu, Tengkai and Srinivasan, Venkatesh and Thomo, Alex},
  year      = {2025}
}
```

---

## Contributing

Bug reports & feature requests are welcome—open an issue or PR.
For major changes, start by discussing your idea with the maintainers.

---

*(c) 2025 Tengkai Yu et al. This code is released under an open-source licence; see `LICENSE` for details.*
