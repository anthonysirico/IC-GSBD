# Iterative Classification for Graph-Set-Based Design

The methodological framework, Iterative Classification for Graph-Set-Based Design (IC-GSBD) employs an iterative approach to efficiently narrow down a graph-based dataset containing diverse design solutions to identify the most useful options.
Utilizing Geometric Deep Learning (GDL) through [PyTorch Geometric](https://github.com/pyg-team/pytorch_geometric), we analyze a small subset of the dataset to train a machine learning model which is then used to predict the remainder of the dataset iteratively, progressively narrowing down to the top solutions.

## Data
The current data consists of a MATLAB `.m`[^Fn1] file that contains over 43,000 graphs that represent Analog Electric Circuits. 
Each circuit contains an adjacency matrix, node labels representing the various components in each circuit, and a graph-level label representing the performance of the circuit.[^Fn2]

## Required Python Libraries
Most of the libraries used are pre-built into Python, but please ensure that you have the following:
- [SciPy](https://scipy.org/)
- [NumPy](https://numpy.org/)
- [Pandas](https://pandas.pydata.org/)
- [tqdm](https://github.com/tqdm/tqdm)

The primary machine learning tools required to be installed are:
- [PyTorch](https://pytorch.org/)
- [PyTorch Geometric](https://pytorch-geometric.readthedocs.io/en/latest/)
- [NetworkX](https://networkx.org/)

## Running the Script
Once the files are downloaded, the `run.py` file is the primary script where you can adjust particular variables such as save paths.

## Future Data
This project is still ongoing, and more engineering architecture data will become available in the near future.

## Cite
Please cite my [dissertation](https://arxiv.org/abs/2303.09770) (and the respective papers of the methods used) if you use this code in your own work:


```
@phdthesis{Sirico2024c,
  author    = {Sirico Jr, Anthony},
  title     = {Integrating geometric deep learning with a set-based design approach for the exploration of graph-based engineering systems},
  type      = {{Ph.D.} {Dissertation}},
  school    = {Colorado State University},
  address   = {Fort Collins, CO, USA},
  month     = aug,
  year      = {2024},
  pdf       = {https://www.engr.colostate.edu/%7Edrherber/files/Sirico2024c.pdf},
}

```

If you notice anything unexpected, please open an [issue](https://github.com/anthonysirico/GDL-for-Engineering-Design/issues) and let us know.
If you have any questions, feel free [to discuss them with us](https://github.com/anthonysirico/GDL-for-Engineering-Design/discussions).

[^Fn1]: MATLAB is not required to run this script. But, if you wish to manipulate any of the raw data, MATLAB or Octave is required.
[^Fn2]: D. R. Herber, T. Guo, J. T. Allison. 'Enumeration of architectures with perfect matchings.' ASME Journal of Mechanical Design, 139(5), p. 051403, May 2017. doi: 10.1115/1.4036132
