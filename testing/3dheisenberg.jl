using TensorNetworkQuantumSimulator
const TN = TensorNetworkQuantumSimulator

using ITensors: ITensors, @OpName_str, @SiteType_str, Algorithm, datatype

using NamedGraphs: edges, NamedGraphs
using Graphs
const NG = NamedGraphs
const G = Graphs
using NamedGraphs.NamedGraphGenerators: named_grid, named_hexagonal_lattice_graph
using NamedGraphs.GraphsExtensions: add_edges, add_vertices

using Random
using TOML

using Base.Threads
using LinearAlgebra

using Adapt
using Dictionaries

BLAS.set_num_threads(min(6, Sys.CPU_THREADS))
println("Julia is using " * string(nthreads()))
println("BLAS is using " * string(BLAS.get_num_threads()))

#TODO: put right mult by X,Y,Z in the library
#Gate : rho -> rho .X in the Pauli basis. With this defined, expect(sqrt_rho, X; alg) = Tr(sqrt_rho . X sqrt_rho) / Tr(sqrt_rho sqrt_rho) = Tr(rho . X) / Tr(rho)
function ITensors.op(
        ::OpName"X", ::SiteType"Pauli"
    )
    mat = zeros(ComplexF64, 4, 4)
    mat[1, 2] = 1
    mat[2, 1] = 1
    mat[3, 4] = im
    mat[4, 3] = -im
    return mat
end

function main()

    #If you want GPU, do rho = CUDA.cu(rho) before measuring or sampling
    #Also you need: using CUDA and a GPU in your computer.

    n = 4
    g = named_grid((n, n, n); periodic = true)
    #Pauli inds are d = 4 and run over I, X, Y, Z
    s = siteinds("Pauli", g)
    ρ = identitytensornetworkstate(ComplexF64, g, s)
    ρ_bpc = TN.update(TN.BeliefPropagationCache(ρ))
    ITensors.disable_warn_order()

    #Near criticality (3.044 is T = 0, critical point)
    δβ = 0.01
    J = 1

    #Imaginary rotations by half angles (as we are targeting sqrt(rho))
    layer = []
    ec = edge_color(g, 6)
    for colored_edges in ec
        append!(layer, ("Rxx", pair, -im * 0.5 * J * δβ) for pair in colored_edges)
        append!(layer, ("Ryy", pair, -im * 0.5 * J * δβ) for pair in colored_edges)
        append!(layer, ("Rzz", pair, -im * 0.5 * J * δβ) for pair in colored_edges)
    end

    @assert length(reduce(vcat, ec)) == length(edges(g))
    nsteps = 50
    apply_kwargs = (; maxdim = 2, cutoff = 1.0e-12)

    β = 0
    for i in 1:nsteps
        ρ_bpc, errs = apply_gates(layer, ρ_bpc; apply_kwargs)

        β += δβ

        ρ = network(ρ_bpc)
        sxx_double_beta_bp = TN.expect(ρ_bpc, ("XX", [(2, 2, 2), (2, 2, 3)]))
        syy_double_beta_bp = TN.expect(ρ_bpc, ("YY", [(2, 2, 2), (2, 2, 3)]))
        szz_double_beta_bp = TN.expect(ρ_bpc, ("ZZ", [(2, 2, 2), (2, 2, 3)]))
        e = sxx_double_beta_bp + syy_double_beta_bp + szz_double_beta_bp


        println("Built sqrt(rho) at inverse Temperature $β")
        println("Bond dimension of PEPO $(TN.maxvirtualdim(ρ))")
        println("Expectation value of X at β  = $(2 * β) is $(sx_double_beta_bp) with BP")
        println("Expectation value of energy at β  = $(2 * β) is $(e) with BP")
    end


    return
end

main()
