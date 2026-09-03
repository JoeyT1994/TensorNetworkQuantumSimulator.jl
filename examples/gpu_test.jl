using TensorNetworkQuantumSimulator
using CUDA
using Base: Base, summarysize
using TensorNetworkQuantumSimulator: inds, TensorNetworkQuantumSimulator
using TensorKit: dim
function main()
    g = named_comb_tree((3,3))

    nqubits = length(vertices(g))
    ψ0 = random_tensornetworkstate(ComplexF32, g, "S=1/2"; bond_dimension = 500)
    ψ0 = CUDA.cu(ψ0)

    maxdim, cutoff = 500, nothing
    apply_kwargs = (; maxdim, cutoff, normalize_tensors = true)

    ψ_bpc = BeliefPropagationCache(ψ0)
    h, J = -2.0, -1.0
    no_trotter_steps = 100
    δt = 0.25

    #Do a 7-way edge coloring then Trotterise the Hamiltonian into commuting groups
    layer = []
    ec = edge_color(g, 3)
    append!(layer, ("Rz", [v], h * δt) for v in vertices(g))
    for colored_edges in ec
        append!(layer, ("Rxx", pair, 2 * J * δt) for pair in colored_edges)
    end
    append!(layer, ("Rz", [v], h * δt) for v in vertices(g))

    #Vertices to measure "Z" on
    vs_measure = [first(center(g))]
    observables = [("Z", [v]) for v in vs_measure]


    χinit = maxvirtualdim(ψ_bpc)
    println("Initial bond dimension of the state is $χinit")

    is = inds(network(ψ_bpc)[only(vs_measure)])
    d = prod(TensorNetworkQuantumSimulator.dim.(is))
    println("Central Factor is $(d*8 / 1e9) GB")

    ψ_bpc = TensorNetworkQuantumSimulator.update(ψ_bpc)
    println("Updated")
    expect_sigmaz = real.(expect(ψ_bpc, observables))
    println("Initial Sigma Z on selected sites is $expect_sigmaz")

    time = 0

    @show [degree(g,v) for v in vertices(g)]
    Zs = []

    # evolve! The first evaluation will take significantly longer because of compilation.
    for l in 1:no_trotter_steps
        #printing
        println("Layer $l")

        # pass BP cache manually
        t = @timed ψ_bpc, errors =
            apply_gates!(layer, ψ_bpc; apply_kwargs, update_cache = false, verbose = false)

        # push BP measured expectation to list
        push!(Zs, only(real(expect(ψ_bpc, observables))))
        
        #println("Cache is $mem GB in memory")
        #mem_f = summarysize(network(ψ_bpc)[only(vs_measure)])/ 1e9
        #println("Central factor is $mem_f GB in memory")
        # printing
        println("Took time: $(t.time) [s]. Max bond dimension: $(maxvirtualdim(ψ_bpc))")
        println("Maximum Gate error for layer was $(maximum(errors))")
        println("Sigma z on central site is $(last(Zs))")
    end
    return
end

main()
