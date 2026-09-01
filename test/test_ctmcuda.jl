@eval module $(gensym())
using Adapt: adapt
using CUDA
using Logging: NullLogger, with_logger
using Random
using TensorNetworkQuantumSimulator
using Test: @testset, @test, @test_skip
const TNQS = TensorNetworkQuantumSimulator

@testset "finite CTMRG CUDA" begin
    if !CUDA.functional()
        @test_skip CUDA.functional()
    else
        CUDA.allowscalar(false)
        Random.seed!(731)
        graph = named_grid((3, 3))
        sites = siteinds("S=1/2", graph)
        cpu = gauge_and_scale(random_tensornetworkstate(
            ComplexF32, graph, sites; bond_dimension = 3))
        gpu = adapt(CUDA.CuArray, cpu)

        # chi=8 gives 72-wide D=3 interfaces, exercising the grouped host-factorization
        # route rather than only cuSOLVER's <=32 exact batched SVD.
        cpu_cache, gpu_cache = with_logger(NullLogger()) do
            cpu_result = TNQS.update(CTMEnvironmentCache(cpu, 8; projector = :cut);
                                     maxiter = 2, tolerance = 0.0)
            gpu_result = TNQS.update(CTMEnvironmentCache(gpu, 8; projector = :cut);
                                     maxiter = 2, tolerance = 0.0)
            (cpu_result, gpu_result)
        end
        cpu_F = cvm_freenergy(cpu_cache)
        gpu_F = cvm_freenergy(gpu_cache)
        @test isfinite(gpu_F)
        @test abs(cpu_F - gpu_F) < 1.0f-4
    end
end
end
