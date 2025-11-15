using TensorNetworkQuantumSimulator
using Test

@testset "TensorNetworkQuantumSimulator.jl" begin
    include("test_constructors.jl")
    include("test_forms.jl")
    include("test_expect.jl")
    include("test_boundarymps.jl")
    include("test_beliefpropagation.jl")
    include("test_sampling.jl")
end
