# Geometric Deep Learning for Engineering Design

Geometric Deep Learning (GDL) is a project that utilizes [PyG](https://github.com/pyg-team/pytorch_geometric) towards graph-based engineering designs.

<img src="Data/drawing.svg" width="800" height="400">

It consists of a method that is capable of taking graph-based architectures to include both node-level and graph-level features for the purposes of architecture down-selection through iterative classification.

## Data
The current data consists of a MATLAB `.m` file that contains over 43,000 graphs that represent Analog Electric Circuits. 
Each circuit contains an adjacency matrix, node labels representing the various components in each circuit, and a graph-level label representing the performance of the circuit.[^Fn1]

Please cite our [conference paper](https://arxiv.org/abs/2303.09770) (and the respective papers of the methods used) if you use this code in your own work:


```
@Inproceedings{SiricoX1,
  title                    = {On the use of geometric deep learning towards the evaluation of graph-centric engineering systems},
  Author                   = {Sirico Jr., Anthony and Herber, Daniel R},
  Booktitle                = {(to appear) ASME 2023 International Design Engineering Technical Conferences},
  Month                    = aug,
  Year                     = {2023},
  Address                  = {Boston, MA, USA},
}
```

If you notice anything unexpected, please open an [issue](https://github.com/anthonysirico/GDL-for-Engineering-Design/issues) and let us know.
If you have any questions, feel free [to discuss them with us](https://github.com/anthonysirico/GDL-for-Engineering-Design/discussions).

[^Fn1]: D. R. Herber, T. Guo, J. T. Allison. 'Enumeration of architectures with perfect matchings.' ASME Journal of Mechanical Design, 139(5), p. 051403, May 2017. doi: 10.1115/1.4036132
