from setuptools import setup, find_packages

setup(
    name="vlouvain",
    version="1.0.0",
    author="Tengkai Yu, Venkatesh Srinivasan, Alex Thomo",
    author_email="yutengkai@uvic.ca",
    description="Vector-Based Louvain Algorithm for Massive Dense Graphs",
    long_description=open("README.md").read(),
    long_description_content_type="text/markdown",
    url="https://github.com/yutengkai/VLouvain",
    packages=find_packages(),
    classifiers=[
        "Development Status :: 4 - Beta",
        "Intended Audience :: Science/Research",
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "Topic :: Scientific/Engineering :: Information Analysis",
    ],
    python_requires=">=3.8",
    install_requires=[
        "torch>=2.0.0",
        "numpy>=1.21.0",
        "tqdm>=4.60.0",
    ],
    extras_require={
        "notebooks": [
            "scikit-learn>=1.0.0",
            "pandas>=1.3.0",
            "torch-geometric>=2.3.0",
            "networkx>=2.6.0",
            "jupyter>=1.0.0",
        ],
    },
)
