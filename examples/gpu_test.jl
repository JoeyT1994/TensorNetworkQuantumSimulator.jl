using TensorNetworkQuantumSimulator
using CUDA

const TNQS = TensorNetworkQuantumSimulator

function main()
    g = named_comb_tree((3, 3))
    bond_dimension = isempty(ARGS) ? 500 : parse(Int, ARGS[1])
    no_trotter_steps = length(ARGS) < 2 ? 100 : parse(Int, ARGS[2])

    ψ0 = random_tensornetworkstate(
        ComplexF32, g, "S=1/2"; bond_dimension,
    )
    ψ0 = CUDA.cu(ψ0)

    maxdim, cutoff = bond_dimension, nothing
    apply_kwargs = (; maxdim, cutoff, normalize_tensors = true)

    ψ_bpc = BeliefPropagationCache(ψ0)
    h, J = -2.0, -1.0
    δt = 0.25

    #Trotterise the Hamiltonian into commuting groups.
    layer = Any[]
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

    is = TNQS.inds(network(ψ_bpc)[only(vs_measure)])
    d = prod(TNQS.dim.(is))
    println("Central Factor is $(d*8 / 1e9) GB")

    ψ_bpc = TNQS.update(ψ_bpc)
    println("Updated")
    expect_sigmaz = real.(expect(ψ_bpc, observables))
    println("Initial Sigma Z on selected sites is $expect_sigmaz")

    @show [degree(g, v) for v in vertices(g)]
    Zs = Float64[]

    # evolve! The first evaluation will take significantly longer because of compilation.
    for l in 1:no_trotter_steps
        println("Layer $l")

        # pass BP cache manually
        t = @timed ψ_bpc, errors =
            apply_gates!(layer, ψ_bpc; apply_kwargs, update_cache = false, verbose = false)

        push!(Zs, only(real(expect(ψ_bpc, observables))))

        println("Took time: $(t.time) [s]. Max bond dimension: $(maxvirtualdim(ψ_bpc))")
        println("Maximum Gate error for layer was $(maximum(errors))")
        println("Sigma z on central site is $(last(Zs))")
    end
    return nothing
end

main()
