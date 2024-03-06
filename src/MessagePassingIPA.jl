module MessagePassingIPA

using Flux: Flux, Dense, Chain, flatten, unsqueeze, chunk, batched_mul, batched_vec, batched_transpose, softplus, sigmoid, relu
using GraphNeuralNetworks: GNNGraph, apply_edges, softmax_edge_neighbors, aggregate_neighbors
using LinearAlgebra: normalize
using Statistics: mean

# Algorithm 21 (x1: N, x2: Ca, x3: C)
function rigid_from_3points(x1::AbstractVector, x2::AbstractVector, x3::AbstractVector)
    v1 = x3 - x2
    v2 = x1 - x2
    e1 = normalize(v1)
    u2 = v2 - e1 * (e1'v2)
    e2 = normalize(u2)
    e3 = e1 × e2
    R = [e1 e2 e3]
    t = x2
    return R, t
end

function rigid_from_3points(x1::AbstractMatrix, x2::AbstractMatrix, x3::AbstractMatrix)
    v1 = x3 .- x2
    v2 = x1 .- x2
    e1 = v1 ./ sqrt.(sum(abs2, v1, dims=1))
    u2 = v2 .- e1 .* sum(e1 .* v2, dims=1)
    e2 = u2 ./ sqrt.(sum(abs2, u2, dims=1))
    e3 = similar(e1)
    e3[1, :] = e1[2, :] .* e2[3, :] .- e1[3, :] .* e2[2, :]
    e3[2, :] = e1[3, :] .* e2[1, :] .- e1[1, :] .* e2[3, :]
    e3[3, :] = e1[1, :] .* e2[2, :] .- e1[2, :] .* e2[1, :]
    R = similar(e1, (3, 3, size(e1, 2)))
    R[:, 1, :] = e1
    R[:, 2, :] = e2
    R[:, 3, :] = e3
    t = x2
    return R, t
end


# Rigid transformation
# --------------------

# N: number of residues
struct RigidTransformation{T,A<:AbstractArray{T,3},B<:AbstractArray{T,2}}
    rotations::A     # (3, 3, N)
    translations::B  # (3, N)
end

"""
    RigidTransformation(rotations, translations)

Create a sequence of rigid transformations.

# Arguments
- `rotations`: 3×3xN array, `rotations[:,:,j]` represents a single rotation
- `translations`: 3×N array, `translations[:,j]` represents a single translation
"""
RigidTransformation

Flux.@functor RigidTransformation

# x: (3, ?, N)
"""
    transform(rigid::RigidTransformation, x::AbstractArray)

Apply transformation `rigid` to `x`.
"""
transform(rigid::RigidTransformation{T}, x::AbstractArray{T,3}) where {T} =
    batched_mul(rigid.rotations, x) .+ unsqueeze(rigid.translations, dims=2)

# y: (3, ?, N)
"""
    inverse_transform(rigid::RigidTransformation, y::AbstractArray)

Apply inverse transformation `rigid` to `y`.
"""
inverse_transform(rigid::RigidTransformation{T}, y::AbstractArray{T,3}) where {T} =
    batched_mul(
        batched_transpose(rigid.rotations),
        y .- unsqueeze(rigid.translations, dims=2),
    )

"""
    compose(rigid1::RigidTransformation, rigid2::RigidTransformation)

Compose two rigid transformations.
"""
function compose(
    rigid1::RigidTransformation{T},
    rigid2::RigidTransformation{T},
) where {T}
    rotations = batched_mul(rigid1.rotations, rigid2.rotations)
    translations =
        batched_vec(rigid1.rotations, rigid2.translations) + rigid1.translations
    return RigidTransformation(rotations, translations)
end


# Invariant point attention
# -------------------------

struct InvariantPointAttention
    # hyperparameters
    n_heads::Int
    c::Int
    n_query_points::Int
    n_point_values::Int

    # trainable layers and weights
    map_nodes::Dense
    map_points::Dense
    map_pairs::Dense
    map_final::Dense
    header_weights_raw::Any
end

Flux.@functor InvariantPointAttention

"""
    InvariantPointAttention(
        n_dims_s, n_dims_z;
        n_heads = 12,
        c = 16,
        n_query_points = 4,
        n_point_values = 8)

Create an invariant point attention layer.
"""
function InvariantPointAttention(
    n_dims_s::Integer,
    n_dims_z::Integer;
    n_heads::Integer=12,
    c::Integer=16,
    n_query_points::Integer=4,
    n_point_values::Integer=8
)
    # initialize layer weights so that outputs have std = 1 (as assumed in
    # AlphaFold2) if inputs follow the standard normal distribution
    init = Flux.kaiming_uniform(gain=1.0)
    map_nodes = Dense(n_dims_s => n_heads * c * 3, bias=false; init)
    map_points = Dense(
        n_dims_s => n_heads * (n_query_points * 2 + n_point_values) * 3,
        bias=false;
        init
    )
    map_pairs = Dense(n_dims_z => n_heads, bias=false; init)
    map_final =
        Dense(n_heads * (n_dims_z + c + n_point_values * (3 + 1)) => n_dims_s, bias=true)
    header_weights_raw = @. log(expm1($(ones(Float32, n_heads))))  # initialized so that initial weights are ones
    return InvariantPointAttention(
        n_heads,
        c,
        n_query_points,
        n_point_values,
        map_nodes,
        map_points,
        map_pairs,
        map_final,
        header_weights_raw,
    )
end

# Algorithm 22
function (ipa::InvariantPointAttention)(
    g::GNNGraph,
    s::AbstractMatrix,
    z::AbstractMatrix,
    rigid::RigidTransformation,
)
    F = eltype(s)
    n_residues = size(s, 2)
    (; n_heads, c, n_query_points, n_point_values) = ipa

    # map inputs (residues come at the last dimension)
    nodes = reshape(ipa.map_nodes(s), n_heads, :, n_residues)
    points = transform(rigid, reshape(ipa.map_points(s), 3, :, n_residues))
    bias = ipa.map_pairs(z)

    # split into queries, keys and values
    # NOTE: workaround to avoid bugs associated with the chunk function
    #nodes_q, nodes_k, nodes_v = chunk(nodes, size=[c, c, c], dims=2)
    i = firstindex(nodes, 2)
    nodes_q = nodes[:,i:i+c-1,:]; i += size(nodes_q, 2)
    nodes_k = nodes[:,i:i+c-1,:]; i += size(nodes_k, 2)
    nodes_v = nodes[:,i:i+c-1,:]
    #points_q, points_k, points_v = chunk(
    #    points,
    #    size=n_heads * [n_query_points, n_query_points, n_point_values],
    #    dims=2,
    #)
    i = firstindex(points, 2)
    points_q = points[:,i:i+n_heads*n_query_points-1,:]; i += size(points_q, 2)
    points_k = points[:,i:i+n_heads*n_query_points-1,:]; i += size(points_k, 2)
    points_v = points[:,i:i+n_heads*n_point_values-1,:]

    points_q = reshape(points_q, 3, n_heads, :, n_residues)
    points_k = reshape(points_k, 3, n_heads, :, n_residues)
    points_v = reshape(points_v, 3, n_heads, :, n_residues)

    # run message passing
    w_C = F(√(2 / 9n_query_points))
    w_L = F(1 / √3)
    γ = softplus.(ipa.header_weights_raw)
    function message(xi, xj, e)
        u = sumdrop(xi.nodes_q .* xj.nodes_k, dims=2)  # inner products
        v = sumdrop(abs2.(xi.points_q .- xj.points_k), dims=(1, 3))  # sum of squared distances
        attn_logits = @. w_L * (1 / √$(F(c)) * u + e - γ * w_C / 2 * v)  # logits of attention scores
        return (; attn_logits, nodes_v=xj.nodes_v, points_v=xj.points_v)
    end
    xi = xj = (; nodes_q, nodes_k, nodes_v, points_q, points_k, points_v)
    e = bias
    msgs = apply_edges(message, g; xi, xj, e)

    # aggregate messages from neighbors
    attn = softmax_edge_neighbors(g, msgs.attn_logits)  # (heads, edges)
    out_pairs =
        aggregate_neighbors(g, +, reshape(attn, n_heads, 1, :) .* unsqueeze(z, dims=1))
    out_nodes = aggregate_neighbors(g, +, reshape(attn, n_heads, 1, :) .* msgs.nodes_v)
    out_points = aggregate_neighbors(g, +, reshape(attn, 1, n_heads, 1, :) .* msgs.points_v)
    out_points = inverse_transform(rigid, reshape(out_points, 3, :, n_residues))
    out_points_norm = sqrt.(sumdrop(abs2.(out_points), dims=1))

    # return the final output
    out = vcat(flatten.((out_pairs, out_nodes, out_points, out_points_norm))...)
    return ipa.map_final(out)
end

sumdrop(x; dims) = dropdims(sum(x; dims); dims)


# Geometric vector perceptron
# ---------------------------

struct GeometricVectorPerceptron
    W_h::AbstractMatrix
    W_μ::AbstractMatrix
    scalar::Dense
    sσ::Function
    vσ::Function
    vgate::Union{Dense, Nothing}
end

Flux.@functor GeometricVectorPerceptron

"""
    GeometricVectorPerceptron(
        (sin, vin) => (sout, vout),
        (sσ, vσ) = (identity, identity);
        bias = true,
        vector_gate = false
    )

Create a geometric vector perceptron layer.

This layer takes a pair of scalar and vector feature arrays that have the size
of `sin × batchsize` and `3 × vin × batchsize`, respectively, and returns a pair
of scalar and vector feature arrays that have the size of `sout × batchsize` and
`3 × vout × batchsize`, respectively. The scalar features are invariant whereas
the vector features are equivariant under any rotation and reflection.

# Arguments
- `sin`, `vin`: scalar and vector input dimensions
- `sout`, `vout`: scalar and vector output dimensions
- `sσ`, `vσ`: scalar and vector nonlinearlities
- `bias`: includes a bias term iff `bias = true`
- `vector_gate`: includes vector gating iff `vector_gate = true`

# References
- Jing, Bowen, et al. "Learning from protein structure with geometric vector perceptrons." arXiv preprint arXiv:2009.01411 (2020).
- Jing, Bowen, et al. "Equivariant graph neural networks for 3d macromolecular structure." arXiv preprint arXiv:2106.03843 (2021).
"""
function GeometricVectorPerceptron(
    ((sin, vin), (sout, vout)),
    (sσ, vσ) = (identity, identity);
    bias::Bool = true,
    vector_gate::Bool = false,
    init = Flux.glorot_uniform
)
    h = max(vin, vout)  # intermediate dimension for vector mapping
    W_h = init(vin, h)
    W_μ = init(h, vout)
    scalar = Dense(sin + h => sout; bias, init)
    vgate = nothing
    if vector_gate
        vgate = Dense(sout => vout, sigmoid; init)
    end
    GeometricVectorPerceptron(W_h, W_μ, scalar, sσ, vσ, vgate)
end

# s: scalar features (sin × batch)
# V: vector feautres (3 × vin × batch)
function (gvp::GeometricVectorPerceptron)(s::AbstractArray{T, 2}, V::AbstractArray{T, 3}) where T
    @assert size(V, 1) == 3
    V_h = batched_mul(V, gvp.W_h)
    s_m = gvp.scalar(cat(norm1drop(V_h), s, dims = 1))
    V_μ = batched_mul(V_h, gvp.W_μ)
    s′ = gvp.sσ.(s_m)
    if gvp.vgate === nothing
        V′ = gvp.vσ.(unsqueeze(norm1drop(V_μ), dims = 1)) .* V_μ
    else
        V′ = unsqueeze(gvp.vgate(gvp.vσ.(s_m)), dims = 1) .* V_μ
    end
    s′, V′
end

# This makes chaining by Flux's Chain easier.
(gvp::GeometricVectorPerceptron)((s, V)::Tuple{AbstractArray{T, 2}, AbstractArray{T, 3}}) where T  = gvp(s, V)

struct GeometricVectorPerceptronGNN
    gvpstack::Chain
end

"""
    GeometricVectorPerceptronGNN(
        (sn, vn),
        (se, ve),
        (sσ, vσ) = (relu, relu);
        n_hidden_layers = 1,
        vector_gate = false,
    )

Create a graph neural network with geometric vector perceptrons.

This layer first concatenates the node and the edge features and then propagates
them over the graph. It returns a pair of scalr and vector feature arrays that
have the same size of input node features.

# Arguments
- `sn`, `vn`: scalar and vector dimensions of node features
- `se`, `ve`: scalar and vector dimensions of edge features
- `sσ`, `sσ`: scalar and vector nonlinearlities
- `vector_gate`: includes vector gating iff `vector_gate = true`
- `n_intermediate_layers`: number of intermediate layers between the input and the output geometric vector perceptrons
"""
function GeometricVectorPerceptronGNN(
    (sn, vn)::Tuple{Integer, Integer},
    (se, ve)::Tuple{Integer, Integer},
    (sσ, vσ)::Tuple{Function, Function} = (relu, relu);
    vector_gate::Bool = false,
    n_intermediate_layers::Integer = 1,
)
    gvpstack = Chain(
        # input layer
        GeometricVectorPerceptron((sn + se, vn + ve) => (sn, vn), (sσ, vσ); vector_gate),
        # intermediate layers
        [
            GeometricVectorPerceptron((sn, vn) => (sn, vn), (sσ, vσ); vector_gate)
            for _ in 1:n_intermediate_layers
        ]...,
        # output layers
        GeometricVectorPerceptron((sn, vn) => (sn, vn)),
    )
    GeometricVectorPerceptronGNN(gvpstack)
end

function (gnn::GeometricVectorPerceptronGNN)(
    g::GNNGraph,
    (sn, vn)::Tuple{<:AbstractArray{T, 2}, <:AbstractArray{T, 3}},
    (se, ve)::Tuple{<:AbstractArray{T, 2}, <:AbstractArray{T, 3}},
) where T
    # run message passing
    function message(_, xj, e)
        s = cat(xj.s, e.s, dims = 1)
        v = cat(xj.v, e.v, dims = 2)
        gnn.gvpstack((s, v))
    end
    xj = (s = sn, v = vn)
    e = (s = se, v = ve)
    msgs = apply_edges(message, g; xj, e)
    aggregate_neighbors(g, mean, msgs)  # return (s, v)
end

# Normalization for vector features
struct VectorNorm
    ϵ::Float32
end

VectorNorm(; eps::Real = 1f-5) = VectorNorm(eps)

function (norm::VectorNorm)(V::AbstractArray{T, 3}) where T
    @assert size(V, 1) == 3
    V ./ (sqrt.(mean(sum(abs2, V, dims = 1), dims = 2)) .+ norm.ϵ)
end

# L2 norm along the first dimension
norm1(X) = sqrt.(sum(abs2, X, dims = 1))
norm1drop(X) = dropdims(norm1(X), dims = 1)

end
