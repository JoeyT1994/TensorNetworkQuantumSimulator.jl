using TensorNetworkQuantumSimulator
const TN = TensorNetworkQuantumSimulator

using ITensorNetworks

using NamedGraphs.NamedGraphGenerators: named_grid
using Statistics

function main()
    nx = 5
    ny = 5

    # the graph is your main friend in working with the TNs
    g = named_grid((nx, ny))
    nq = length(vertices(g))

    dt = 0.25

    hx = 1.0
    hz = 0.8
    J = 0.5

    # pauli rotations are tuples like `(pauli_string, [site_labels], parameter)`
    layer = []
    append!(layer, ("Rx", [v], 2*hx*dt) for v in vertices(g))
    append!(layer, ("Rz", [v], 2*hz*dt) for v in vertices(g))
    append!(layer, ("Rzz", pair, 2*J*dt) for pair in edges(g));

    # observables are tuples like `(pauli_string, [site_labels], optional:coefficient)`
    # it's important that the `site_labels` match the names of the vertices of the graph `g`
    obs = ("Z", [(3, 3)])  # right in the middle

    # the number of circuit layers
    nl = 20

    # the initial state
    ψ = zerostate(g)

    # an array to keep track of expectations
    expectations = Float64[real(expect(ψ, obs))]

    # max bond dimension for the TN
    # we will use enough and just see how
    apply_kwargs = (maxdim = 5, cutoff = 1e-10, normalize = false)

    # evolve! The first evaluation will take significantly longer because of compilation.
    for l = 1:nl
        #printing
        println("Layer $l")

        # apply layer
        t = @timed ψ, errors = apply(layer, ψ; apply_kwargs);

        # push expectation to list
        push!(expectations, real(expect(ψ, obs)))

        # printing
        println("    Took time: $(t.time) [s]. Max bond dimension: $(maxlinkdim(ψ))")
        println("    Maximum Gate error for layer was $(maximum(errors))")
    end


    ## A few more advanced options
    # we will still do exactly the same evolution but also do boundary mps for expectation values

    # these kwargs are used every time the BP is updated, but you can pass other kwargs to individual functions 
    set_global_bp_update_kwargs!(maxiter = 25, tol = 1e-6)
    set_global_boundarymps_update_kwargs!(
        message_update_kwargs = (; niters = 20, tolerance = 1e-10),
    )

    # the initial state
    ψ = zerostate(g)

    # create the BP cache manually
    ψψ = build_bp_cache(ψ)

    # an array to keep track of expectations
    expectations_advanced = Float64[real(expect(ψ, obs))]
    boundarymps_rank = 4

    # evolve! The first evaluation will take significantly longer because of compulation.
    for l = 1:nl
        println("Layer $l")

        # pass BP cache manually
        t1 = @timed ψ, ψψ, errors =
            apply(layer, ψ, ψψ; apply_kwargs, verbose = false);

        ## could also update outside 
        # t2 = @timed ψψ = updatecache(ψψ)

        # push expectation to list
        # pass the cache instead of the state so that things don't have to update over and over
        push!(
            expectations_advanced,
            real(
                expect(
                    ψ,
                    obs;
                    alg = "boundarymps",
                    cache_construction_kwargs = (; message_rank = boundarymps_rank),
                ),
            ),
        )  # with some boundary mps correction


        println("    Took time: $(t1.time) [s]. Max bond dimension: $(maxlinkdim(ψ))")
        println("    Maximum Gate error for layer was $(maximum(errors))")
    end
end

main()
