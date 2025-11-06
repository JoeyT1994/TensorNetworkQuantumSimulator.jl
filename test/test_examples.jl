@eval module $(gensym())
using TensorNetworkQuantumSimulator: TensorNetworkQuantumSimulator
using Test: @testset

@testset "Test examples" begin
    example_files = [
        "boundarymps.jl", "heavyhexIsing_dynamics.jl", "loopcorrections.jl", "2dIsing_dynamics.jl", "3dIsing_dynamics.jl",
        "2dIsing_dynamics_Heisenbergpicture.jl"
    ]
    @testset "Test $example_file" for example_file in example_files
        include(joinpath(pkgdir(TensorNetworkQuantumSimulator), "examples", example_file))
    end
end
end
