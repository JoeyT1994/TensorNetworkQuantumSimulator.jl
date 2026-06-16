# --- Measurement: pick a bond (1:z) or a site (:A/:B) ------------------------
#
# Gate application happens like everywhere else in the library (simple update),
# but measuring an InfiniteTensorNetworkState requires picking a contraction
# backend. Only "bp" is supported right now, and the user must ask for it
# explicitly (mirroring `default_alg` for a bare TensorNetworkState).

default_alg(itns::InfiniteTensorNetworkState) = error(
    "You must specify a contraction algorithm. Currently only alg = \"bp\" is supported for an InfiniteTensorNetworkState."
)

function _itns_algorithm_check(alg)
    alg == "bp" && return nothing
    return error(
        "Unsupported algorithm \"$alg\" for an InfiniteTensorNetworkState. Currently only alg = \"bp\" is supported."
    )
end

"""
    iTNS_reduced_density_matrix(ψ_bpc::BeliefPropagationCache, loc; kwargs...)
    iTNS_reduced_density_matrix(ψ::InfiniteTensorNetworkState, loc; alg, kwargs...)

RDM on a single site `loc::Symbol` (`:A`/`:B`), or the two-site RDM across bond
`loc::Int` (sites `:A`,`:B` with bond `loc` explicit and all other bonds traced
into the environments). Given an `InfiniteTensorNetworkState` you must pass an
`alg` (currently only `alg = "bp"` is supported); given a `BeliefPropagationCache`
the backend is already fixed.
"""
iTNS_reduced_density_matrix(ψ_bpc::BeliefPropagationCache, loc::Union{Symbol, Int}; kwargs...) =
    reduced_density_matrix(ψ_bpc, _locverts(loc); kwargs...)

function iTNS_reduced_density_matrix(
        itns::InfiniteTensorNetworkState, loc::Union{Symbol, Int};
        alg::Union{String, Nothing} = default_alg(itns),
        bp_update_kwargs = default_bp_update_kwargs(itns.tns), kwargs...
    )
    _itns_algorithm_check(alg)
    ψ_bpc = update(BeliefPropagationCache(itns.tns); bp_update_kwargs...)
    return iTNS_reduced_density_matrix(ψ_bpc, loc; kwargs...)
end

"""
    iTNS_expect(ψ_bpc::BeliefPropagationCache, op, loc; kwargs...)
    iTNS_expect(ψ::InfiniteTensorNetworkState, op, loc; alg, kwargs...)

Expectation value of `op` on a site `loc::Symbol` (e.g. `iTNS_expect(ψ, "Z", :A)`)
or across bond `loc::Int` (e.g. `iTNS_expect(ψ, "ZZ", 1)`). Given an
`InfiniteTensorNetworkState` you must pass an `alg` (currently only `alg = "bp"`
is supported); given a `BeliefPropagationCache` the backend is already fixed.
"""
function iTNS_expect(ψ_bpc::BeliefPropagationCache, op, loc::Union{Symbol, Int}; kwargs...)
    # Fermionic networks measure through the validated parity-aware `expect` (doubled network
    # with Jordan–Wigner operator strings), not via a reduced density matrix. `op` is given as
    # operator name(s) understood by the fermionic backend, e.g. `"N"`/`"Nup"` on a site, or a
    # pair like `["Cdag", "C"]` across a bond.
    if is_fermionic(network(ψ_bpc))
        op_strings = op isa AbstractString ? String[op] : collect(String, op)
        # `_obsverts(loc)` (= [A, B] for a bond) is only the operator SUPPORT and does not name
        # the bond; A and B are joined by `z` parallel bond vertices, so the contraction region
        # is ambiguous and the Steiner tree would silently pick the same bond every time. Pass
        # `_locverts(loc)` (= [A, bond_k, B]) as the explicit cluster so bond `k` is the one
        # contracted exactly and the other bonds enter through their messages — i.e. the result
        # actually depends on `loc`.
        return expect(ψ_bpc, (op_strings, _obsverts(loc)); alg = "bp", contract_vertices = _locverts(loc))
    end
    ρ = iTNS_reduced_density_matrix(ψ_bpc, loc; kwargs...)
    O = _resolve_gate(op, loc, graph(ψ_bpc), siteinds(network(ψ_bpc)))
    return (ρ * O)[]
end

function iTNS_expect(
        itns::InfiniteTensorNetworkState, op, loc::Union{Symbol, Int};
        alg::Union{String, Nothing} = default_alg(itns),
        bp_update_kwargs = default_bp_update_kwargs(itns.tns), kwargs...
    )
    _itns_algorithm_check(alg)
    ψ_bpc = update(BeliefPropagationCache(itns.tns); bp_update_kwargs...)
    return iTNS_expect(ψ_bpc, op, loc; kwargs...)
end
