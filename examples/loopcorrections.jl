using TensorNetworkQuantumSimulator
const TN = TensorNetworkQuantumSimulator

using ITensorNetworks
const ITN = ITensorNetworks
using ITensors

using NamedGraphs
using Graphs
const NG = NamedGraphs
const G = Graphs
using NamedGraphs.NamedGraphGenerators: named_grid, named_hexagonal_lattice_graph

using EinExprs: Greedy

using Random
Random.seed!(1634)

function main()
    nx, ny = 5, 5
    χ = 3
    ITensors.disable_warn_order()
    gs = [
        (named_grid((nx, 1)), "line", 0),
        (named_hexagonal_lattice_graph(nx, ny), "hexagonal", 6),
        (named_grid((nx, ny)), "square", 4),
    ]
    for (g, g_str, smallest_loop_size) in gs
        println("Testing for $g_str lattice with $(NG.nv(g)) vertices")
        s = siteinds("S=1/2", g)
        ψ = ITN.random_tensornetwork(ComplexF64, s; link_space = χ)
        s = ITN.siteinds(ψ)

        ψ = normalize(ψ; alg = "bp")

        norm_sqr_bp = inner(ψ, ψ; alg = "loopcorrections", max_configuration_size = 0, cache_update_kwargs = TN.default_posdef_bp_update_kwargs())
        norm_sqr = inner(
            ψ,
            ψ;
            alg = "loopcorrections",
            max_configuration_size = 2*(smallest_loop_size) - 1,
        )
        norm_sqr_exact = inner(
            ψ,
            ψ;
            alg = "exact",
            contraction_sequence_kwargs = (; alg = "einexpr", optimizer = Greedy()),
        )

        println("Bp Value for norm is $norm_sqr_bp")
        println("1st Order Loop Corrected Value for norm is $norm_sqr")
        println("Exact Value for norm is $norm_sqr_exact")
    end
end

main()
