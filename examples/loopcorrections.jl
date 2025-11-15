using TensorNetworkQuantumSimulator

using ITensors

using LinearAlgebra: norm

using EinExprs: Greedy

using Random
Random.seed!(1634)

function main()
    nx, ny = 4, 4
    χ = 3
    ITensors.disable_warn_order()
    gs = [
        (named_grid((nx, 1)), "line", 0),
        (named_hexagonal_lattice_graph(nx, ny), "hexagonal", 6),
        (named_grid((nx, ny)), "square", 4),
    ]
    for (g, g_str, smallest_loop_size) in gs
        println("Testing for $g_str lattice with $(nv(g)) vertices")
        ψ = random_tensornetworkstate(ComplexF32, g, "S=1/2"; bond_dimension = χ)

        ψ = normalize(ψ; alg = "bp")

        norm_bp = norm(ψ; alg = "bp")
        norm_loopcorrected = norm(ψ; alg = "loopcorrections", max_configuration_size = 2 * (smallest_loop_size) - 1)
        norm_exact = norm(ψ; alg = "exact")

        println("Bp Value for norm is $norm_bp")
        println("1st Order Loop Corrected Value for norm is $norm_loopcorrected")
        println("Exact Value for norm is $norm_exact")
    end
    return
end

main()
