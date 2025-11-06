using TensorNetworkQuantumSimulator
const TN = TensorNetworkQuantumSimulator

using ITensors

using NamedGraphs
using Graphs
const NG = NamedGraphs
const G = Graphs
using NamedGraphs.NamedGraphGenerators: named_grid

function main()
    nx, ny = 4, 4
    g = named_grid((nx, ny))

    nqubits = length(vertices(g))
    #Physical indices represent "Identity, X, Y, Z" in that order
    vz = first(center(g))
    ψ0 = paulitensornetworkstate(ComplexF32, v -> v == vz ? "Z" : "I", g)

    maxdim, cutoff = 4, 1.0e-14
    apply_kwargs = (; maxdim, cutoff, normalize_tensors = false)
    #Parameters for BP, as the graph is not a tree (it has loops), we need to specify these

    ψ = copy(ψ0)

    ψ_bpc = BeliefPropagationCache(ψ)

    h, J = -1.0, -1.0
    no_trotter_steps = 10
    δt = 0.04

    #Do a 4-way edge coloring then Trotterise the Hamiltonian into commuting groups. Lets do Ising with the designated parameters
    layer = []
    ec = edge_color(g, 4)
    append!(layer, ("Rz", [v], h * δt) for v in vertices(g))
    for colored_edges in ec
        append!(layer, ("Rxx", pair, 2 * J * δt) for pair in colored_edges)
    end
    append!(layer, ("Rz", [v], h * δt) for v in vertices(g))

    χinit = maxvirtualdim(ψ)
    println("Initial bond dimension of the Heisenberg operator is $χinit")

    time = 0

    Zs = Float64[]

    for l in 1:no_trotter_steps
        println("Layer $l")

        #Apply the circuit
        t = @timed ψ_bpc, errors =
            apply_gates(layer, ψ_bpc; apply_kwargs, verbose = false)
        #Reset the Frobenius norm to unity
        ψ_bpc = TN.rescale(ψ_bpc)
        println("Frobenius norm of O(t) is $(TN.partitionfunction(ψ_bpc))")

        ψ = network(ψ_bpc)
        #Take traces
        tr_ψt = inner(ψ, TN.identitytensornetworkstate(g, siteinds(ψ)); alg = "bp")
        tr_ψtψ0 = inner(ψ, ψ0; alg = "bp")
        println("Trace(O(t)) is $(tr_ψt)")
        println("Trace(O(t)O(0)) is $(tr_ψtψ0)")

        # printing
        println("Took time: $(t.time) [s]. Max bond dimension: $(maxvirtualdim(ψ_bpc))")
        println("Maximum Gate error for layer was $(maximum(errors))")
    end
    return
end

main()