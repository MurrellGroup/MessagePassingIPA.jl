using MessagePassingIPA: RigidTransformation, InvariantPointAttention, rigid_from_3points
using GraphNeuralNetworks: rand_graph
using BatchedTransformations
using Test

@testset "MessagePassingIPA.jl" begin
    @testset "RigidTransformation" begin
        n = 100
        rigid = rand(Float32, Rigid, 3, (n,))
        x = randn(Float32, 3, 12, n)
        y = rigid * x
        @test size(x) == size(y)
        @test x ≈ inverse(rigid) * y

        n = 100
        rigid1 = rand(Float32, Rigid, 3, (n,))
        rigid2 = rand(Float32, Rigid, 3, (n,))
        rigid12 = rigid1 ∘ rigid2
        x = randn(Float32, 3, 12, n)
        @test rigid12 * x ≈ rigid1 * (rigid2 * x)
        y = rigid12 * x
        @test x ≈ inverse(rigid2) * (inverse(rigid1) * y)
    end

    @testset "InvariantPointAttention" begin
        n_dims_s = 32
        n_dims_z = 16
        ipa = InvariantPointAttention(n_dims_s, n_dims_z)

        n_nodes = 100
        n_edges = 500
        g = rand_graph(n_nodes, n_edges)
        s = randn(Float32, n_dims_s, n_nodes)
        z = randn(Float32, n_dims_z, n_edges)

        # check returned type and size
        c = randn(Float32, 3, n_nodes) * 1000
        x1 = c .+ randn(Float32, 3, n_nodes)
        x2 = c .+ randn(Float32, 3, n_nodes)
        x3 = c .+ randn(Float32, 3, n_nodes)
        rigid1 = RigidTransformation(rigid_from_3points(x1, x2, x3)...)
        @test ipa(g, s, z, rigid1) isa Matrix{Float32}
        @test size(ipa(g, s, z, rigid1)) == (n_dims_s, n_nodes)

        # check invariance
        R, t = values(rand(Float32, Rotation, 3)), randn(Float32, 3)
        x1 = R * x1 .+ t
        x2 = R * x2 .+ t
        x3 = R * x3 .+ t
        rigid2 = RigidTransformation(rigid_from_3points(x1, x2, x3)...)
        @test ipa(g, s, z, rigid1) ≈ ipa(g, s, z, rigid2)
    end
end
