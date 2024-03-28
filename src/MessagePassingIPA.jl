module MessagePassingIPA

using Flux: Flux, Dense, flatten, unsqueeze, chunk, batched_mul, batched_vec, batched_transpose, softplus, sigmoid
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

nresidues(rigid::RigidTransformation) = size(rigid.translations, 2)

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

transform(rigid::RigidTransformation{T}, x::AbstractArray{T, 4}) where T =
    reshape(transform(rigid, reshape(x, 3, :, nresidues(rigid))), size(x))

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

inverse_transform(rigid::RigidTransformation{T}, x::AbstractArray{T, 4}) where T =
    reshape(inverse_transform(rigid, reshape(x, 3, :, nresidues(rigid))), size(x))

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
    nodes_q, nodes_k, nodes_v = chunk(nodes, size=[c, c, c], dims=2)
    points_q, points_k, points_v = chunk(
        points,
        size=n_heads * [n_query_points, n_query_points, n_point_values],
        dims=2,
    )

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


# Invariant point gate
# --------------------

struct MultiHeadGate{A <: AbstractMatrix, B <: AbstractVector, F <: Function}
    U::A
    V::A
    W::A
    b::B
    σ::F
end

Flux.@layer MultiHeadGate

function MultiHeadGate(
    n_dims_x::Integer,
    n_dims_z::Integer,
    n_heads::Integer,
    σ::Function,
)
    init = Flux.kaiming_normal()
    U = init(n_heads, n_dims_x)
    V = init(n_heads, n_dims_x)
    W = init(n_heads, n_dims_z)
    b = zeros(Float32, n_heads)
    MultiHeadGate(U, V, W, b, σ)
end

function (gate::MultiHeadGate)(si, sj, zij)
    (; U, V, W, b, σ) = gate
    σ.(U*si .+ V*sj .+ W*zij .+ b)
end

struct InvariantPointGate
    # hyperparameters
    n_heads::Int
    n_point_values::Int

    # trainable layers
    map_points::Dense
    map_final::Dense
    gate::MultiHeadGate
end

Flux.@layer InvariantPointGate

function InvariantPointGate(
    n_dims_s::Integer,
    n_dims_z::Integer;
    n_heads::Integer = 24,
    n_point_values::Integer = 14,
    σ::Function = sigmoid,
)
    init = Flux.kaiming_uniform(gain = 1.0)
    map_points = Dense(n_dims_s => 3 * n_point_values * n_heads; bias = false, init)
    map_final = Dense(3 * n_point_values * n_heads => n_dims_s; bias = true, init)
    gate = MultiHeadGate(n_dims_s, n_dims_z, n_heads, σ)
    InvariantPointGate(
        n_heads,
        n_point_values,
        map_points,
        map_final,
        gate,
    )
end

function (ipg::InvariantPointGate)(
    g::GNNGraph,
    s::AbstractMatrix,
    z::AbstractMatrix,
    rigid::RigidTransformation
)
    (; n_heads, n_point_values, gate) = ipg
    points = transform(rigid, reshape(ipg.map_points(s), 3, n_point_values, n_heads, :))
    function message(xi, xj, zij)
        # transform xj points to the local frames of xi
        rigid = RigidTransformation(xi.rotations, xi.translations)
        points = inverse_transform(rigid, xj.points)
        reshape(gate(xi.s, xj.s, zij), 1, 1, n_heads, :) .* points
    end
    xi = xj = (; s, points, rotations = rigid.rotations, translations = rigid.translations)
    msgs = apply_edges(message, g; xi, xj, e = z)
    out_points = aggregate_neighbors(g, +, msgs)
    ipg.map_final(flatten(out_points))
end

sumdrop(x; dims) = dropdims(sum(x; dims); dims)

end
