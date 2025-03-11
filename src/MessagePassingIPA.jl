module MessagePassingIPA

using Flux: Flux, Dense, flatten, unsqueeze, chunk, batched_mul, batched_vec, batched_transpose, softplus
using GraphNeuralNetworks: GNNGraph, apply_edges, softmax_edge_neighbors, aggregate_neighbors
using LinearAlgebra: normalize
using BatchedTransformations

# Algorithm 21 (x1: N, x2: Ca, x3: C)
function rigid_from_3points(x1::AbstractVector, x2::AbstractVector, x3::AbstractVector)
    v1 = x3 - x2
    v2 = x1 - x2
    e1 = normalize(v1)
    u2 = v2 - e1 * (e1'v2)
    e2 = normalize(u2)
    e3 = e1 × e2
    R = [e1 e2 e3]
    t = reshape(x2, 3, 1)
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
    t = reshape(x2, 3, 1, :)
    return R, t
end

function RigidTransformation(R::AbstractArray{T,3}, t::AbstractArray{T,3}) where T<:Real
    Translation(t) ∘ Rotation(R)
end

RigidTransformation(R, t::AbstractMatrix) = RigidTransformation(R, reshape(t, 3, 1, :))

# Invariant point attention
# -------------------------

struct InvariantPointAttention
    # hyperparameters
    n_heads::Int
    c::Int
    n_query_points::Int
    n_point_values::Int

    # trainable layers and weights
    map_nodes::NamedTuple
    map_points::NamedTuple
    map_pairs::Dense
    map_final::Dense
    header_weights_raw::Any
end

Flux.@layer InvariantPointAttention

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
    map_nodes = (
        q = Dense(n_dims_s => n_heads * c, bias=false; init),
        k = Dense(n_dims_s => n_heads * c, bias=false; init),
        v = Dense(n_dims_s => n_heads * c, bias=false; init),
    )
    map_points = (
        q = Dense(n_dims_s => n_heads * n_query_points * 3, bias=false; init),
        k = Dense(n_dims_s => n_heads * n_query_points * 3, bias=false; init),
        v = Dense(n_dims_s => n_heads * n_point_values * 3, bias=false; init),
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
    rigid::Rigid,
)
    F = eltype(s)
    n_residues = size(s, 2)
    (; n_heads, c, n_query_points, n_point_values) = ipa

    # map inputs (residues come at the last dimension)
    nodes_q = reshape(ipa.map_nodes.q(s), n_heads, c, n_residues)
    nodes_k = reshape(ipa.map_nodes.k(s), n_heads, c, n_residues)
    nodes_v = reshape(ipa.map_nodes.v(s), n_heads, c, n_residues)
    points_q = reshape(transform(rigid, reshape(ipa.map_points.q(s), 3, :, n_residues)), 3, n_heads, n_query_points, n_residues)
    points_k = reshape(transform(rigid, reshape(ipa.map_points.k(s), 3, :, n_residues)), 3, n_heads, n_query_points, n_residues)
    points_v = reshape(transform(rigid, reshape(ipa.map_points.v(s), 3, :, n_residues)), 3, n_heads, n_point_values, n_residues)
    bias = ipa.map_pairs(z)

    # run message passing
    w_C = F(√(2 / 9n_query_points))
    w_L = F(1 / √3)
    γ = softplus.(ipa.header_weights_raw)
    function message(xi, xj, e)
        u = sumdrop(xi.nodes_q .* xj.nodes_k, dims=2)  # inner products
        v = sumdrop(abs2, xi.points_q .- xj.points_k, dims=(1, 3))  # sum of squared distances
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
    out_points_norm = sqrt.(sumdrop(abs2, out_points, dims=1))

    # return the final output
    out = vcat(flatten.((out_pairs, out_nodes, out_points, out_points_norm))...)
    return ipa.map_final(out)
end

sumdrop(f, x; dims) = dropdims(sum(f, x; dims); dims)
sumdrop(x; dims) = sumdrop(identity, x; dims)

end
