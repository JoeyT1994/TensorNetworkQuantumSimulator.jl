using TensorNetworkQuantumSimulator
using Test

@testset "TensorNetworkQuantumSimulator.jl" begin
    include("test_constructors.jl")
    include("test_forms.jl")
    include("test_expect.jl")
    include("test_boundarymps.jl")
    include("test_beliefpropagation.jl")
    include("test_apply.jl")
    include("test_sampling.jl")
    include("test_truncate.jl")
    include("test_loop_cluster_graph.jl")
    include("test_cluster_expand.jl")
end
