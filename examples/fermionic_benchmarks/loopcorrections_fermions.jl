using TensorNetworkQuantumSimulator

using ITensors

using LinearAlgebra: norm

using Random
Random.seed!(1634)

function main()
    nx, ny = 4,4
    χ = 3
    ITensors.disable_warn_order()
    gs = [
        (named_grid((nx, 1)), "line", 0),
        (named_hexagonal_lattice_graph(nx, ny), "hexagonal", 6),
        (named_grid((nx, ny)), "square", 4),
    ]
    for (g, g_str, smallest_loop_size) in gs
        s = siteinds("spinful_fermion", g)
        max_configuration_size = 2*smallest_loop_size -1
        println("\n")
        println("-----------------------")
        obs = (["Nup"], [first(center(g))])
        println("Testing for $g_str lattice with $(nv(g)) vertices")
        ψ = random_fermionic_tensornetworkstate(ComplexF64, g, s; bond_dimension = χ)

        ψ = normalize(ψ; alg = "bp")

        norm_bp = norm(ψ; alg = "bp")
        norm_loopcorrected = norm(ψ; alg = "loopcorrections", max_configuration_size)
        norm_exact = norm(ψ; alg = "exact")

        println("Bp Value for norm is $norm_bp")
        println("1st Order Loop Corrected Value for norm is $norm_loopcorrected")
        println("Exact Value for norm is $norm_exact")

        nupdn_bp = expect(ψ, obs; alg = "bp")
        nupdn_loopcorrected = expect(ψ, obs; alg = "loopcorrections",max_configuration_size)
        nupdn_exact = expect(ψ, obs; alg = "exact")

        println("\n")

        println("Bp Value for nupdn is $nupdn_bp")
        println("1st Order Loop Corrected Value for nupdn is $nupdn_loopcorrected")
        println("Exact Value for nupdn is $nupdn_exact")
    end
    return
end

main()
