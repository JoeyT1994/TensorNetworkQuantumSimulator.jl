@eval module $(gensym())
using Test: @test, @testset
include(joinpath(@__DIR__, "mpi_runner.jl"))

@testset "Test BP MPI cache" begin
    @test run_mpi_worker("path", 2)
    @test run_mpi_worker("ring", 2)
    @test run_mpi_worker("chain3", 3)
end

@testset "Test BP MPI apply" begin
    @test run_mpi_worker("apply_path", 2)
    @test run_mpi_worker("apply_ring", 2)
end

# Partitions built per rank, with no rank ever holding the global network.
@testset "Test BP MPI apply, locally built partitions" begin
    @test run_mpi_worker("localbuild_path", 2)
    @test run_mpi_worker("localbuild_ring", 2)
end
end
