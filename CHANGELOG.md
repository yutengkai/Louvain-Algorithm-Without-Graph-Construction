# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/).

## [1.0.0] - 2026-03-24

### Added
- LICENSE file (MIT)
- Proper requirements.txt with pinned versions
- setup.py for pip installation
- CONTRIBUTING.md guidelines
- CHANGELOG.md (this file)
- Unit tests for core algorithm
- Colab badges for notebooks in README

### Changed
- Completely rewrote README.md for better readability
- Updated citation with EDBT 2026 DOI (10.48786/edbt.2026.43)
- Renamed `Paper_Notebook_Homogeneous_IGprah.ipynb` to fix typo → `Paper_Notebook_Homogeneous_IGraph.ipynb`

### Removed
- Empty test_preprocessing.py placeholder (not needed)

### Fixed
- Fixed paper title in README (was "Dense Graphs", now correctly "Low-Rank Graphs")

## [0.1.0] - 2025-06-03

### Added
- Initial release with core VLouvain algorithm
- Support for positive and negative edge weights
- Experiment notebooks comparing against cuGraph, iGraph, NetworKit, GVE
- PyTorch Geometric dataset loading helpers
