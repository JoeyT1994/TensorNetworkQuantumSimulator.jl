using CUDA
using LinearAlgebra
using Printf
using Random

function batched_svd(matrices, workers)
    output = Vector{Any}(undef, length(matrices))
    CUDA.synchronize()
    start = time_ns()
    @sync for worker in 1:workers
        @async begin
            for index in worker:workers:length(matrices)
                output[index] = svd(copy(matrices[index]))
            end
            CUDA.synchronize()
            nothing
        end
    end
    return output, (time_ns() - start) / 1.0e9
end

function main()
    CUDA.functional() || error("CUDA is not functional")
    CUDA.allowscalar(false)
    Random.seed!(1234)
    count = parse(Int, get(ENV, "CTM_GPU_SVD_COUNT", "48"))
    size_ = parse(Int, get(ENV, "CTM_GPU_SVD_SIZE", "72"))
    host = [randn(ComplexF32, size_, size_) for _ in 1:count]
    matrices = CuArray.(host)
    for workers in (1, 2, 4, 8)
        batched_svd(matrices[1:min(workers, count)], workers) # handles + kernels
        output, elapsed = batched_svd(matrices, workers)
        error = maximum(norm(host[i] - Array(output[i].U) * Diagonal(Array(output[i].S)) *
                             Array(output[i].Vt)) / norm(host[i]) for i in eachindex(host))
        @printf("workers=%d seconds=%.6f max_residual=%.3e\n", workers, elapsed, error)
    end
end

main()
