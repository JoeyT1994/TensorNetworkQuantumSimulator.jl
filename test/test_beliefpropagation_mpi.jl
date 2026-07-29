@eval module $(gensym())
using Test: @test, @testset
include(joinpath(@__DIR__, "mpi_runner.jl"))

# Grouped by rank count, not by topic: each mpiexec launch reloads the package on every rank, so
# one call per group is several times faster than one per case. Case names are the keys of CASES
# in mpi_beliefpropagation_worker.jl, and the worker prints a per-case verdict.
@testset "Test BP MPI, 2 ranks" begin
    @test run_mpi_worker(
        [
            # Cache construction, boundary bookkeeping and BP against a serial reference.
            "path", "ring",
            # Gate application, both with a pre-built global state and with partitions each rank
            # builds itself (no rank ever holding the whole network).
            "apply_path", "apply_ring", "localbuild_path", "localbuild_ring",
            # The public entry point: a tuple circuit spanning the graph, plus distributed expect.
            "apply_gates_mpi_path", "apply_gates_mpi_ring",
            # Ranks must agree on the sweep count without being told it.
            "defaults_path", "defaults_ring",
            # A malformed partition must throw on every rank rather than hang.
            "validation",
        ],
        2
    )
end

@testset "Test BP MPI, 3 ranks" begin
    @test run_mpi_worker(["chain3", "defaults_chain3"], 3)
end
end
