module MessagePassingIPA

using Flux: Flux, Dense, flatten, unsqueeze, chunk, batched_mul, batched_vec, batched_transpose, softplus
using GraphNeuralNetworks: GNNGraph, apply_edges, softmax_edge_neighbors, aggregate_neighbors
using LinearAlgebra: normalize

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

struct GeometricVectorPerceptron
    W_h
    W_μ
    scalar::Dense
    vσ
end

function GeometricVectorPerceptron((sin, sout), (vin, vout), sσ::Function = identity, vσ::Function = identity; bias = true)
    h = max(vin, vout)  # intermediate dimension for vector mapping
    W_h = randn(Float32, vin, h)
    W_μ = randn(Float32, h, vout)
    scalar = Dense(sin + h => sout, sσ; bias)
    GeometricVectorPerceptron(W_h, W_μ, scalar, vσ)
end

function (gvp::GeometricVectorPerceptron)(s::AbstractArray, V::AbstractArray)
    V_h = batched_mul(V, gvp.W_h)
    s′ = gvp.scalar(cat(norm1(V_h), s, dims = 1))
    V_μ = batched_mul(V_h, gvp.W_μ)
    V′ = gvp.vσ(unsqueeze(norm1(V_μ), dims = 1)) .* V_μ
    s′, V′
end

# L2 norm along the first dimension
norm1(X) = dropdims(sqrt.(sum(abs2.(X), dims = 1)), dims = 1)

end
