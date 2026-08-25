using TensorNetworkQuantumSimulator
const TNQS = TensorNetworkQuantumSimulator
using TensorNetworkQuantumSimulator: update
using Random
using CUDA
function main()
    Random.seed!(1234)
    #g = named_hexagonal_lattice_graph(4,4)
    g = named_grid((4,4))
    s = siteinds("S=1/2", g)
    ψ = random_tensornetworkstate(ComplexF64, g,s; bond_dimension = 4)
    ψ = gauge_and_scale(ψ)

    obs = ("Z", (1,1))
    Z_exact = norm_sqr(ψ; alg = "exact")
    O_exact = expect(ψ, obs; alg = "exact")

    projector = :cycle
    Rs= [1,2,4,8, 16]
    for R in Rs
        @show R
        ψ_ctm_cpu = CTMEnvironmentCache(ψ, R; projector)
        t1= time()
        ψ_ctm_cpu = update(ψ_ctm_cpu; maxiter = 100, tolerance = 1e-14, verbose = false)
        t2 = time()
        println("Update took $(t2 - t1) secs on CPU")
        err_ctmrg = (log(Z_exact) - TNQS.cvm_freenergy(ψ_ctm_cpu))
        err_bmps = (log(Z_exact) - log(norm_sqr(ψ; alg = "boundarymps", mps_bond_dimension = R)))
        println("Testing norms")
        @show err_ctmrg
        @show err_bmps

        println("Testing Obs")
        err_bmps = abs(O_exact - expect(ψ, obs; alg = "boundarymps", mps_bond_dimension = R))
        err_ctmrg = abs(O_exact - expect(ψ_ctm_cpu, obs))
        @show err_ctmrg
        @show err_bmps
    end
end

main()
