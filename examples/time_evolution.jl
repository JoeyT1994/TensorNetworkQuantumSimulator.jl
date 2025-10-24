using TensorNetworkQuantumSimulator
const TN = TensorNetworkQuantumSimulator

using NamedGraphs.NamedGraphGenerators: named_grid
using Statistics

function main()
    nx = 5
    ny = 5

    # the graph is your main friend. This will be the geometry of the TN you wull work with
    g = named_grid((nx, ny))
    nq = length(vertices(g))

    dt = 0.25

    hx = 1.0
    hz = 0.8
    J = 0.5

    #Build a layer of the circuit. Pauli rotations are tuples like `(pauli_string, [site_labels], parameter)`
    layer = []
    append!(layer, ("Rx", [v], 2*hx*dt) for v in vertices(g))
    append!(layer, ("Rz", [v], 2*hz*dt) for v in vertices(g))

    #For two site gates do an edge coloring to Trotterise the circuit
    ec = edge_color(g, 4)
    for colored_edges in ec
        append!(layer, ("Rzz", pair, 2*J*dt) for pair in colored_edges)
    end

    # observables are tuples like `(pauli_string, [site_labels], optional:coefficient)`
    # it's important that the `site_labels` match the names of the vertices of the graph `g`
    obs = ("Z", [(3, 3)])  # right in the middle

    # the number of circuit layers
    nl = 20

    # the initial state (all up, use Float 32 precision)
    ψ0 = tensornetworkstate(ComplexF32, v -> "↑", g, "S=1/2")

    # max bond dimension for the TN
    apply_kwargs = (maxdim = 5, cutoff = 1e-10, normalize_tensors = false)

    # create the BP cache representing the square of the tensor network
    ψ_bpc = BeliefPropagationCache(ψ0)

    # an array to keep track of expectations taken via two methods
    #expectations_boundarymps = [real(expect(ψψ, obs))]
    #expectations_bp = [real(expect(ψ0, obs))]

    mps_bond_dimension = 4

    # evolve! (First step takes long due to compilation)
    for l = 1:nl
        println("Layer $l")

        t1 = @timed ψ_bpc, errors =
            apply_gates(layer, ψ_bpc; apply_kwargs, verbose = false);

        #BP expectation (already have an up-to-date BP cache)
        sz_bp = expect(ψ_bpc, obs)

        #Boundary MPS expectation 
        ψ = network(ψ_bpc)
        sz_boundarymps = expect(ψ,obs;alg = "boundarymps", mps_bond_dimension)

        println("    Took time: $(t1.time) [s]. Max bond dimension: $(maxlinkdim(ψ_bpc))")
        println("    Maximum Gate error for layer was $(maximum(errors))")

        println("    BP Measured Sigmaz is $(sz_bp)")
        println("    Boundary MPS Measured Sigmaz is $(sz_boundarymps)")
    end
end

main()
