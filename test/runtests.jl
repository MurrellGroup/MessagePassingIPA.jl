using MessagePassingIPA
using GraphNeuralNetworks: rand_graph
using Rotations: RotMatrix
using Test

@testset "MessagePassingIPA.jl" begin
    @testset "RigidTransformation" begin
        n = 100
        rotations = stack(rand(RotMatrix{3,Float32}) for _ in 1:n)
        translations = randn(Float32, 3, n)
        rigid = MessagePassingIPA.RigidTransformation(rotations, translations)
        x = randn(Float32, 3, 12, n)
        y = MessagePassingIPA.transform(rigid, x)
        @test size(x) == size(y)
        @test x ≈ MessagePassingIPA.inverse_transform(rigid, y)
    end

    @testset "InvariantPointAttention" begin
        n_dims_s = 32
        n_dims_z = 16
        ipa = MessagePassingIPA.InvariantPointAttention(n_dims_s, n_dims_z)

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
        rigid1 = MessagePassingIPA.RigidTransformation(MessagePassingIPA.rigid_from_3points(x1, x2, x3)...)
        @test ipa(g, s, z, rigid1) isa Matrix{Float32}
        @test size(ipa(g, s, z, rigid1)) == (n_dims_s, n_nodes)

        # check invariance
        R, t = rand(RotMatrix{3,Float32}), randn(Float32, 3)
        x1 = R * x1 .+ t
        x2 = R * x2 .+ t
        x3 = R * x3 .+ t
        rigid2 = MessagePassingIPA.RigidTransformation(MessagePassingIPA.rigid_from_3points(x1, x2, x3)...)
        @test ipa(g, s, z, rigid1) ≈ ipa(g, s, z, rigid2)
    end
end
