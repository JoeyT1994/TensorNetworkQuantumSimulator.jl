function expect(
        alg::Algorithm"exact",
        ψ::TensorNetworkState,
        observables::Vector{<:Tuple};
        contraction_sequence_kwargs = (; alg = "omeinsum", optimizer = GreedyMethod())
    )

    denom = norm_sqr(alg, ψ; contraction_sequence_kwargs)
    out = Number[]
    for obs in observables
        op_strings, vs, coeff = collectobservable(obs, graph(ψ))
        if iszero(coeff)
            push!(out, zero(coeff))
            continue
        end
        op_string_f = op_string_function(op_strings, vs)
        ψOψ_tensors = norm_factors(ψ, collect(vertices(ψ)); op_strings = op_string_f)
        # `norm_factors` returns `nothing` for a parity-forbidden fermionic observable (odd
        # number of parity-odd operator factors); such an expectation is zero by parity.
        if ψOψ_tensors === nothing
            push!(out, zero(coeff))
            continue
        end
        numer_seq = contraction_sequence(ψOψ_tensors; contraction_sequence_kwargs...)
        numer = scalar(contract(ψOψ_tensors; sequence = numer_seq))
        push!(out, coeff * (numer / denom))
    end
    return out
end

function expect(
        alg::Algorithm"exact",
        ψ::TensorNetworkState,
        observable::Tuple;
        kwargs...
    )
    return only(expect(alg, ψ, [observable]; kwargs...))
end

"""
    expect(ψ, observable; alg="exact", kwargs...) -> Number or Vector{Number}

Compute the expectation value of one or more observables on a tensor network state.

# Arguments
- `ψ::Union{TensorNetworkState, BeliefPropagationCache, BoundaryMPSCache}`: The tensor network state or cache wrapping the state to measure the observable(s) on.
- `observable::Union{Tuple, Vector{<:Tuple}}`: The observable(s) to measure. Should be a tuple or vector of tuples of the form `(ops, vertices, coeff=1)`.

# Keyword Arguments
- `alg::Union{String, Nothing}`: The algorithm to use. Supported algorithms:
    - `"exact"`: Exact contraction of the tensor network.
    - `"bp"`: Belief propagation approximation.
    - `"boundarymps"`: Boundary MPS approximation (requires `mps_bond_dimension`).
- `cache_update_kwargs...`: Keyword arguments passed to the `update` function when using `"bp"` or `"boundarymps"` algorithms.

# Returns
- A single number if measuring one observable, or a vector of numbers if measuring multiple observables.
"""
function expect(ψ::Union{TensorNetworkState, BeliefPropagationCache, BoundaryMPSCache}, observable; alg::Union{String, Nothing} = default_alg(ψ), kwargs...)
    algorithm_check(ψ, "expect", alg)
    return expect(Algorithm(alg), ψ, observable; kwargs...)
end

function expect(
        alg::Algorithm"bp",
        cache::BeliefPropagationCache,
        obs::Tuple;
        contract_vertices = nothing,
    )
    op_strings, obs_vs, coeff = collectobservable(obs, graph(cache))
    iszero(coeff) && return zero(coeff)

    # The cluster contracted exactly (everything else enters through BP messages). Normally it
    # is the Steiner tree spanning the operator support. `contract_vertices`, when given, fixes
    # it explicitly — needed when the support alone does not pin the region down, e.g. an iTNS
    # bond where A and B are joined by several PARALLEL bond vertices and the Steiner tree would
    # otherwise always pick the same one (making every bond's expectation identical). It must
    # contain the operator support so the operators still land on their sites.
    steiner_vs = if contract_vertices !== nothing
        all(in(contract_vertices), obs_vs) ||
            error("`contract_vertices` = $(contract_vertices) must contain the observable support $(obs_vs).")
        collect(contract_vertices)
    elseif length(obs_vs) == 1
        obs_vs
    else
        # Canonicalize the support order before choosing the region. `steiner_tree` can pick a
        # different equal-length path depending on the input vertex order, so the bare support
        # would make ⟨O_i O_j⟩ and ⟨O_j O_i⟩ contract over different clusters and break the
        # Hermiticity invariant ⟨c†_i c_j⟩ = conj⟨c†_j c_i⟩ on loopy graphs. Sort by `string`
        # rather than the vertices themselves so this works for any (possibly non-`isless`)
        # vertex label type.
        collect(vertices(steiner_tree(network(cache), sort(obs_vs; by = string))))
    end
    incoming_ms = incoming_messages(cache, steiner_vs)

    #TODO: If there are a lot of tensors here, (more than 100 say), we need to think about defining a custom sequence as optimal may be too slow
    # For fermionic networks `norm_factors` returns `nothing` when the operator string is
    # parity-forbidden (odd total parity ⇒ ⟨O⟩ = 0); propagate that as a `nothing` region.
    function contract_region(op_string_f)
        tensors = norm_factors(network(cache), steiner_vs; op_strings = op_string_f)
        tensors === nothing && return nothing
        append!(tensors, incoming_ms)
        seq = contraction_sequence(tensors; alg = "optimal", prune_tensors = true)
        return scalar(contract(tensors; sequence = seq))
    end

    denom = contract_region(v -> "I")
    numer = contract_region(op_string_function(op_strings, obs_vs))
    numer === nothing && return zero(coeff)

    return coeff * numer / denom
end

function expect(
        alg::Algorithm"boundarymps",
        cache::BoundaryMPSCache,
        obs::Tuple;
        bmps_messages_up_to_date = false,
    )
    op_strings, obs_vs, coeff = collectobservable(obs, graph(cache))
    iszero(coeff) && return zero(coeff)

    op_string_f = op_string_function(op_strings, obs_vs)

    numer, denom = path_contract(cache, obs_vs, op_string_f; bmps_messages_up_to_date)
    # `numer` is a (fermionic or bosonic) scalar tensor, or `0` when the observable is
    # parity-forbidden (odd total parity ⇒ ⟨O⟩ = 0).
    numer isa Tensor || return coeff * numer / denom
    return coeff * scalar(numer) / denom
end

function expect(
        alg::Algorithm"boundarymps",
        cache::BoundaryMPSCache,
        observables::Vector{<:Tuple};
        bmps_messages_up_to_date = false,
        kwargs...,
    )
    obs_vs = observables_vertices(observables, graph(cache))
    if !bmps_messages_up_to_date
        cache = update_partitions(cache, obs_vs)
    end
    out = map(obs -> expect(alg, cache, obs; bmps_messages_up_to_date = true, kwargs...), observables)
    return out
end

function expect(
        alg::Algorithm"bp",
        cache::BeliefPropagationCache,
        observables::Vector{<:Tuple};
        kwargs...,
    )
    return map(obs -> expect(alg, cache, obs; kwargs...), observables)
end

function expect(
        alg::Algorithm"bp",
        ψ::TensorNetworkState,
        observable::Union{Tuple, Vector{<:Tuple}};
        cache_update_kwargs = default_bp_update_kwargs(ψ),
        kwargs...,
    )

    ψ_bpc = BeliefPropagationCache(ψ)
    ψ_bpc = update(ψ_bpc; cache_update_kwargs...)

    return expect(alg, ψ_bpc, observable; kwargs...)
end

function expect(
        alg::Algorithm"boundarymps",
        ψ::TensorNetworkState,
        observable::Union{Tuple, Vector{<:Tuple}};
        cache_update_kwargs = default_bmps_update_kwargs(ψ),
        partition_by = boundarymps_partitioning(observable, graph(ψ)),
        mps_bond_dimension::Integer,
        gauge_state = true,
        kwargs...,
    )

    ψ_bmps = BoundaryMPSCache(ψ, mps_bond_dimension; partition_by, gauge_state)
    cache_update_kwargs = with_default_maxiter(cache_update_kwargs, ψ_bmps)
    ψ_bmps = update(ψ_bmps; cache_update_kwargs...)

    obs_vs = observables_vertices(observable, graph(ψ))
    ψ_bmps = update_partitions(ψ_bmps, obs_vs)

    return expect(alg, ψ_bmps, observable; bmps_messages_up_to_date = true, kwargs...)
end

"""
    expect(ψ, obs; alg = "loopcorrections", max_configuration_size, ε = 1e-4, cache_update_kwargs...)

Free-energy / generating-function estimate of a **single-site Hermitian** observable
`obs = (op, vertex[, coeff])`,

    ⟨Ô⟩ = ∂_ε ln⟨ψ|e^{ε Ô}|ψ⟩|_0  ≈  [F(ε) − F(−ε)] / (2ε),

with `F(t) = ln‖e^{t Ô/2}|ψ⟩‖²` the loop-corrected free energy of that norm network
(`loopcorrected_free_energy`, the additive linked-cluster form `ln Z_BP + Σ_C w_C`;
`max_configuration_size` counts EDGES). BP is re-solved for each shifted
network so the loop corrections include the linear response of the messages.

`ε` is the central finite-difference step (default `1e-4`). Pass `autodiff = true` to instead
take the derivative by forward-mode AD (`ForwardDiff` dual numbers): a single BP re-solve with
a `Dual`-seeded gate gives the exact `∂_ε`, removing the `ε` step-size/truncation tradeoff (at
the cost of losing BLAS on the dual-typed contractions). `Ô` must be Hermitian (the
generating-function identity `⟨ψ|e^{εÔ}|ψ⟩ = ‖e^{εÔ/2}|ψ⟩‖²` relies on `Ô† = Ô`). Accepts a
`TensorNetworkState` or a `BeliefPropagationCache`, and a single observable or a vector of
them; multi-site observables are not supported.
"""
function expect(
        alg::Algorithm"loopcorrections",
        ψ_bpc::BeliefPropagationCache,
        obs::Tuple;
        max_configuration_size::Integer, ε::Real = 1e-4, autodiff::Bool = false,
        cache_update_kwargs = default_bp_update_kwargs(ψ_bpc),
    )
    op_strings, obs_vs, coeff = collectobservable(obs, graph(ψ_bpc))
    iszero(coeff) && return zero(coeff)
    length(obs_vs) == 1 ||
        error("expect(...; alg = \"loopcorrections\") supports single-site observables only; got $(length(obs_vs)) sites.")
    v = only(obs_vs)
    op_string = only(op_strings)

    if autodiff
        # Forward-mode AD: ⟨Ô⟩ = ½ F'(0) with F(α) = ln⟨ψ|e^{2αÔ}|ψ⟩. Seed α with a real
        # `Dual` and read the partial off the (complex) free energy in a SINGLE BP re-solve,
        # giving the exact derivative (no ε step-size / truncation tradeoff). `cache_update_kwargs`
        # is computed on the unperturbed (float) cache, so the BP tolerance stays a concrete Float.
        α = ForwardDiff.Dual(0.0, 1.0)
        F = gated_lc_free_energy(ψ_bpc, op_string, v, α, max_configuration_size; cache_update_kwargs)
        dF = complex(ForwardDiff.partials(real(F))[1], ForwardDiff.partials(imag(F))[1])
        return coeff * dF / 2
    end

    Fp = gated_lc_free_energy(ψ_bpc, op_string, v, +ε / 2, max_configuration_size; cache_update_kwargs)
    Fm = gated_lc_free_energy(ψ_bpc, op_string, v, -ε / 2, max_configuration_size; cache_update_kwargs)
    return coeff * (Fp - Fm) / (2ε)
end

function expect(alg::Algorithm"loopcorrections",
    ψ::TensorNetworkState, obs::Tuple; cache_update_kwargs= default_bp_update_kwargs(ψ), kwargs...)
    ψ_bpc = update(BeliefPropagationCache(ψ); cache_update_kwargs...)
    return expect(alg, ψ_bpc, obs; kwargs...)
end

function expect(
        alg::Algorithm"loopcorrections",
        ψ::Union{TensorNetworkState, BeliefPropagationCache},
        observables::Vector{<:Tuple};
        kwargs...,
    )
    return map(obs -> expect(alg, ψ, obs; kwargs...), observables)
end


#Process an observable into more readable form
function collectobservable(obs::Tuple, g::NamedGraph)

    coeff = length(obs) == 2 ? 1 : last(obs)
    verts = observables_vertices(obs, g)
    op = obs[1]

    length(op) != length(verts) && error("Invalid observable: need as many operators as vertices passed.")
    if op isa String
        op_strings = [string(o) for o in op]
    elseif op isa Vector{<:String}
        op_strings = [o for o in op]
    else
        error("Invalid observable, did not recognize operator specification. Either a single string (one pauli character per vertex) or a vector of strings (one string per vertex) expected.")
    end

    return op_strings, verts, coeff
end

# Map each vertex to its operator string, defaulting to the identity "I" off the observable's support.
function op_string_function(op_strings, vs)
    op_dict = Dict(zip(vs, op_strings))
    return v -> get(op_dict, v, "I")
end

observables_vertices(observable::Tuple, g::NamedGraph) = collect_vertices(observable[2], g)
observables_vertices(observables::Vector{<:Tuple}, g::NamedGraph) = unique(collect(Iterators.flatten(observables_vertices(obs, g) for obs in observables)))

function boundarymps_partitioning(observable::Union{Tuple, Vector{<:Tuple}}, g::NamedGraph)
    observables = observable isa Tuple ? [observable] : observable
    partitioning = nothing
    for o in observables
        vs = observables_vertices(o, g)
        if allequal(first.(vs)) && (partitioning == "row" || partitioning == nothing)
            partitioning = "row"
        elseif allequal(last.(vs)) && (partitioning == "col" || partitioning == nothing)
            partitioning = "col"
        else
            error("Observables must all be aligned in either the same column or the same row to do BoundaryMPS measurements.")
        end
    end
    return partitioning
end
