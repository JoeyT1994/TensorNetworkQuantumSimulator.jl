# --- The BP engine (gate step) ----------------------------------------------

function simple_update_iTNS(
    o::ITensor, ψ::TensorNetworkState, v⃗, dtype; envs, normalize_tensors = true, apply_kwargs...
)
    if length(v⃗) == 1
        updated_tensors = ITensor[ITensors.apply(o, ψ[first(v⃗)])]
        s_values, err = nothing, 0
        old_cind_dim, new_cind_dim = 1,1
    else
        ψ1, ψ2 = ψ[v⃗[1]], replaceind(ψ[v⃗[3]], commonind(ψ[v⃗[3]], ψ[v⃗[2]]), commonind(ψ[v⃗[2]], ψ[v⃗[1]]))
        envs_v1 = filter(env -> hascommoninds(env, ψ1), envs)
        envs_v2 = filter(env -> hascommoninds(env, ψ2), envs)
        @assert all(ndims(env) == 2 for env in vcat(envs_v1, envs_v2))
        old_cind_dim = dim(commonind(ψ1, ψ2))
        sqrt_inv_sqrt_envs_v1 = pseudo_sqrt_inv_sqrt.(envs_v1)
        sqrt_inv_sqrt_envs_v2 = pseudo_sqrt_inv_sqrt.(envs_v2)
        sqrt_envs_v1, inv_sqrt_envs_v1 = first.(sqrt_inv_sqrt_envs_v1), last.(sqrt_inv_sqrt_envs_v1)
        sqrt_envs_v2, inv_sqrt_envs_v2 = first.(sqrt_inv_sqrt_envs_v2), last.(sqrt_inv_sqrt_envs_v2)

        sᵥ₁ = commoninds(ψ1, o)
        sᵥ₂ = commoninds(ψ2, o)

        for sqrt_m in sqrt_envs_v1
            ψ1 = ψ1 * sqrt_m
        end

        for sqrt_m in sqrt_envs_v2
            ψ2 = ψ2 * sqrt_m
        end

        Qᵥ₁, Rᵥ₁ = qr(ψ1, uniqueinds(uniqueinds(ψ1, ψ2), sᵥ₁))
        Qᵥ₂, Rᵥ₂ = qr(ψ2, uniqueinds(uniqueinds(ψ2, ψ1), sᵥ₂))
        rᵥ₁ = commoninds(Qᵥ₁, Rᵥ₁)
        oR = ITensors.apply(o, Rᵥ₁ * Rᵥ₂)
        singular_values! = Ref(ITensor())
        Rᵥ₁, Rᵥ₂, spec = factorize_svd(
            oR,
            unioninds(rᵥ₁, sᵥ₁);
            ortho = "none",
            singular_values!,
            apply_kwargs...,
        )
        err = spec.truncerr
        s_values = singular_values![]
        for inv_sqrt_m in inv_sqrt_envs_v1
            Qᵥ₁ = Qᵥ₁ * dag(inv_sqrt_m)
        end

        for inv_sqrt_m in inv_sqrt_envs_v2
            Qᵥ₂ = Qᵥ₂ * dag(inv_sqrt_m)
        end

        cind = commonind(Rᵥ₁, Rᵥ₂)
        new_cind_dim = dim(cind)
        new_cind = Index(new_cind_dim, "Link")
        middle_tensor = adapt(dtype)(ITensors.denseblocks(ITensors.delta(cind, new_cind)))
        updated_tensors = [Qᵥ₁ * Rᵥ₁, middle_tensor, ITensors.replaceind(Qᵥ₂ * Rᵥ₂, cind, new_cind)]
        if normalize_tensors
            s_values = normalize(s_values)
        end
    end

    if normalize_tensors
        updated_tensors = ITensor[ψᵥ / norm(ψᵥ) for ψᵥ in updated_tensors]
    end

    return noprime.(updated_tensors), s_values, err, old_cind_dim, new_cind_dim
end

function apply_gate_iTNS!(gate, ψ_bpc::BeliefPropagationCache, v⃗, dtype; kwargs...)
    envs = incoming_messages(ψ_bpc, v⃗)
    updated_tensors, s_values, err, old_cind_dim, new_cind_dim = simple_update_iTNS(
        gate, network(ψ_bpc), v⃗, dtype; envs = envs, kwargs...
    )

    for (i, v) in enumerate(v⃗)
        setindex_preserve!(ψ_bpc, updated_tensors[i], v)
    end

    if length(v⃗) == 3
        v1, v2, v3 = v⃗
        e1, e2 = NamedEdge(v1 => v2), NamedEdge(v2 => v3)

        ind2 = commonind(s_values, first(updated_tensors))
        s_values = replaceinds(denseblocks(s_values), inds(s_values), [ind2, ind2'])
        setmessage!(ψ_bpc, e1, dag(s_values))
        setmessage!(ψ_bpc, reverse(e1), s_values)

        ind3 = commonind(updated_tensors[2], last(updated_tensors))
        s_values = replaceinds(s_values, [ind2, ind2'], [ind3, ind3'])
        setmessage!(ψ_bpc, e2, dag(s_values))
        setmessage!(ψ_bpc, reverse(e2), s_values)
    end

    return ψ_bpc, err
end

# --- Gate application: pick a bond (1:z) or a site (:A/:B) --------------------

function _gate_tuple(op, verts)
    op isa Tuple || return (op, verts)
    length(op) == 1 ? (op[1], verts) : (op[1], verts, op[2:end]...)
end

_resolve_gate(gate, loc, g, sinds) =
    gate isa ITensor ? gate : first(toitensor(_gate_tuple(gate, _obsverts(loc)), g, sinds))

"""
    iTNS_apply_gate(ψ_bpc::BeliefPropagationCache, gate, loc; apply_kwargs...)
    iTNS_apply_gate(ψ::InfiniteTensorNetworkState, gate, loc; bp_update_kwargs, apply_kwargs...)

Apply `gate` across bond `loc::Int` (a two-site gate on `:A`,`:B`) or on a single
site `loc::Symbol` (`:A`/`:B`). `gate` is an `ITensor` over the relevant site
indices, or a circuit-tuple spec like `"H"`, `("Rxx", θ)`, `"ZZ"`.
`apply_kwargs` (e.g. `maxdim`, `cutoff`) are forwarded to the SVD truncation.

The cache form mutates a copy and is the right one for a sweep of gates (the
converged messages persist between gates). The state form is a convenience that
builds and converges a fresh cache each call.
"""
function iTNS_apply_gate(ψ_bpc::BeliefPropagationCache, gate, loc::Union{Symbol, Int}; apply_kwargs...)
    ψ_bpc = copy(ψ_bpc)
    O = _resolve_gate(gate, loc, graph(ψ_bpc), siteinds(network(ψ_bpc)))
    return apply_gate_iTNS!(O, ψ_bpc, _locverts(loc), datatype(network(ψ_bpc)); apply_kwargs...)
end

function iTNS_apply_gate(
        itns::InfiniteTensorNetworkState, gate, loc::Union{Symbol, Int};
        bp_update_kwargs = default_bp_update_kwargs(itns.tns), apply_kwargs...
    )
    ψ_bpc = update(BeliefPropagationCache(itns.tns); bp_update_kwargs...)
    ψ_bpc, err = iTNS_apply_gate(ψ_bpc, gate, loc; apply_kwargs...)
    return InfiniteTensorNetworkState(network(ψ_bpc)), err
end
