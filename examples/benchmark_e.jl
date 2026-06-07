using TensorNetworkQuantumSimulator

using TensorNetworkQuantumSimulator: norm_factors, toriccode_groundstate, ising_partitionfunction

function main()
    n = 5
    g = named_grid((n,n); periodic = true)
    β = 0.1
    z = ising_partitionfunction(g, β)

    @show (1/(n*n))*log(contract(z; alg = "bp"))
    @show (1/(n*n))*log(contract(z; alg = "exact"))
end

main()