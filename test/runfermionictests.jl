using TensorNetworkQuantumSimulator
using Test

@testset "TensorNetworkQuantumSimulator.jl (fermions)" begin
    include("test_fermions.jl")
    include("test_fermionic_simple_update.jl")
    include("test_fermionic_factorizations.jl")
    include("test_fermionic_bmps.jl")
end
