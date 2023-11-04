# MessagePassingIPA

[![Build Status](https://github.com/bicycle1885/MessagePassingIPA.jl/actions/workflows/CI.yml/badge.svg?branch=main)](https://github.com/bicycle1885/MessagePassingIPA.jl/actions/workflows/CI.yml?query=branch%3Amain)


This package introduces an Invariant Point Attention (IPA) layer coupled with
graph-based message passing, tailored to process structured data by effectively
leveraging both geometric and topological information for superior
representation learning. The operations within this package are designed to
support automatic differentiation and GPU acceleration, ensuring optimal
performance.

For a deeper understanding, you may refer to [the AlphaFold2
paper](https://doi.org/10.1038/s41586-021-03819-2), particularly Algorithm 22 in
the supplementary material.

Jumper, J., Evans, R., Pritzel, A. et al. Highly accurate protein structure
prediction with AlphaFold. Nature 596, 583–589 (2021).


# Usage

```julia
# Load packages
using MessagePassingIPA: RigidTransformation, InvariantPointAttention, rigid_from_3points
using GraphNeuralNetworks: rand_graph

# Initialize an IPA layer
n_dims_s = 32  # the dimension of single representations
n_dims_z = 16  # the diemnsion of pair representations
ipa = InvariantPointAttention(n_dims_s, n_dims_z)

# Generate a random graph and node/edge features
n_nodes = 100
n_edges = 500
g = rand_graph(n_nodes, n_edges)
s = randn(Float32, n_dims_s, n_nodes)
z = randn(Float32, n_dims_z, n_edges)

# Generate random atom coordinates
p = randn(Float32, 3, n_nodes) * 100  # centroid
x1 = p .+ randn(Float32, 3, n_nodes)  # N atoms
x2 = p .+ randn(Float32, 3, n_nodes)  # CA atoms
x3 = p .+ randn(Float32, 3, n_nodes)  # C atoms
rigid = RigidTransformation(rigid_from_3points(x1, x2, x3)...)

# Apply the IPA layer
out = ipa(g, s, z, rigid)
@assert size(out) == (n_dims_s, n_nodes)
```
