using TensorNetworkQuantumSimulator
using Graphs: center
using TensorNetworkQuantumSimulator: setindex_preserve!, noprime
using TensorNetworkQuantumSimulator: Ops, ITensor

function main()
    nx, ny = 4, 4
    g = named_grid((nx, ny))

    nqubits = length(vertices(g))
    #Two indices per site: s[v][1] is the ket leg, s[v][2] is the bra leg of the operator
    vz = first(center(g))
    s = siteinds("S=1/2", g; inds_per_site = 2)
    #Start from the identity operator, then place a single Z on vertex vz
    ψI = identity_tensornetworkstate(ComplexF64, g, s)
    ψ0 = copy(ψI)
    setindex_preserve!(ψ0, noprime(ψ0[vz] * Ops.op("Z", s[vz][1])), vz)

    maxdim, cutoff = 4, 1.0e-14
    apply_kwargs = (; maxdim, cutoff, normalize_tensors = false)
    #Parameters for BP, as the graph is not a tree (it has loops), we need to specify these

    ψ0 = normalize(ψ0; alg = "bp")
    ψ = copy(ψ0)

    ψ_bpc = BeliefPropagationCache(ψ)

    h, J = -1.0, -1.0
    no_trotter_steps = 10
    δt = 0.04

    #Do a 4-way edge coloring then Trotterise the Hamiltonian into commuting groups. Lets do Ising with the designated parameters
    layer = ITensor[]
    ec = edge_color(g, 4)
    
    #Ket leg (s[v][1]) gets U† (angle negated), bra leg (s[v][2]) gets U (angle unchanged) so that O -> U'OU
    append!(layer, [Ops.op("Rz", s[v][1];  θ = -h * δt)*Ops.op("Rz", s[v][2];  θ = h * δt) for v in vertices(g)])
    for es in ec
        append!(layer, [Ops.op("Rxx", s[src(e)][1], s[dst(e)][1], ϕ = -J * δt)*Ops.op("Rxx", s[src(e)][2], s[dst(e)][2], ϕ = J * δt) for e in es])
    end
    append!(layer, [Ops.op("Rz", s[v][1];  θ = -h * δt)*Ops.op("Rz", s[v][2];  θ = h * δt) for v in vertices(g)])

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
        rescale!(ψ_bpc)
        println("Frobenius norm of O(t) is $(partitionfunction(ψ_bpc))")

        ψ = network(ψ_bpc)
        #Take traces
        tr_ψt = inner(ψ, ψI; alg = "bp")
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
