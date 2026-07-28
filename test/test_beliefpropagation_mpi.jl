@eval module $(gensym())
using Test: @testset, @test
include(joinpath(@__DIR__, "mpi_runner.jl"))

@testset "Test BP MPI cache" begin
    @test run_mpi_worker("path", 2)
    @test run_mpi_worker("ring", 2)
    @test run_mpi_worker("chain3", 3)
end
end
