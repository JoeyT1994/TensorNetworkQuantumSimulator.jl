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

# --- The BP engine (gate step), FERMIONIC ------------------------------------
#
# Locally-ordered (arXiv:2410.02215) analogue of `simple_update_iTNS`. The geometry is the
# same A —(bond k)— ABk —(bond k)— B chain, but every operation respects fermionic arrows:
#   * the bond vertex ABk is MERGED into the B side (ψ2 = ψ[B]·ψ[ABk]) rather than collapsed
#     by `replaceind`; the merge keeps arrows consistent (ABk holds A's bond leg opposite to
#     A), so the gate-region contraction never clashes arrows and the trick survives repeated
#     gates on the same bond;
#   * the gate is applied by the permute-then-ordinary-contract rule (see Apply/simple_update.jl);
#   * the new A–B bond is re-split by inserting a parity-even identity bond-vertex tensor with
#     arrows opposite to each neighbour — transparent (parity-diagonal ⇒ no net supertrace),
#     restoring the explicit ABk so BP keeps one independent message per bond.
function simple_update_iTNS(
        o::FermionicITensor, ψ::TensorNetworkState, v⃗, dtype; envs, normalize_tensors = true, apply_kwargs...
    )
    if length(v⃗) == 1
        updated_tensors = FermionicITensor[noprime(o * ψ[first(v⃗)])]
        s_values, err = nothing, 0
        old_cind_dim, new_cind_dim = 1, 1
    else
        vA, vAB, vB = v⃗[1], v⃗[2], v⃗[3]
        # Merge the bond vertex into B: A and (B·ABk) are then directly joined by A's bond
        # index with consistent arrows.
        ψ1 = ψ[vA]
        ψ2 = ψ[vB] * ψ[vAB]
        old_cind_dim = dim(only(commoninds(ψ1, ψ2)))

        envs_v1 = filter(env -> hascommoninds(env, ψ1), envs)
        envs_v2 = filter(env -> hascommoninds(env, ψ2), envs)
        @assert all(ndims(env) == 2 for env in vcat(envs_v1, envs_v2))

        sqrt_inv_sqrt_envs_v1 = pseudo_sqrt_inv_sqrt.(envs_v1)
        sqrt_inv_sqrt_envs_v2 = pseudo_sqrt_inv_sqrt.(envs_v2)
        sqrt_envs_v1, inv_sqrt_envs_v1 = first.(sqrt_inv_sqrt_envs_v1), last.(sqrt_inv_sqrt_envs_v1)
        sqrt_envs_v2, inv_sqrt_envs_v2 = first.(sqrt_inv_sqrt_envs_v2), last.(sqrt_inv_sqrt_envs_v2)

        for sqrt_m in sqrt_envs_v1
            ψ1 = ψ1 * sqrt_m
        end
        for sqrt_m in sqrt_envs_v2
            ψ2 = ψ2 * sqrt_m
        end

        sᵥ₁ = commoninds(ψ1, o)
        sᵥ₂ = commoninds(ψ2, o)

        Qᵥ₁, Rᵥ₁ = qr(ψ1, collect(Index, uniqueinds(uniqueinds(ψ1, ψ2), sᵥ₁)))
        Qᵥ₂, Rᵥ₂ = qr(ψ2, collect(Index, uniqueinds(uniqueinds(ψ2, ψ1), sᵥ₂)))
        rᵥ₁ = commoninds(Qᵥ₁, Rᵥ₁)

        # Gate application: bring the two physical legs adjacent with a fermionic permute (this
        # threads the Koszul sign through the QR bond lying between them), then contract the
        # gate as an ordinary o ⊗ I product onto those legs.
        RR = Rᵥ₁ * Rᵥ₂
        s1ᵢ, s2ᵢ = only(sᵥ₁), only(sᵥ₂)
        rest = filter(i -> i != s1ᵢ && i != s2ᵢ, RR.order)
        RRadj = ITensors.permute(RR, Index[s1ᵢ, s2ᵢ, rest...])
        oR = FermionicITensor(
            noprime(o.tensor * RRadj.tensor),
            copy(RRadj.order), copy(RRadj.dirs), RRadj.grading,
        )
        Rᵥ₁, Rᵥ₂, s_values, err = symmetric_svd(
            oR, collect(Index, unioninds(rᵥ₁, sᵥ₁)); apply_kwargs...
        )

        for inv_sqrt_m in inv_sqrt_envs_v1
            Qᵥ₁ = Qᵥ₁ * dag(inv_sqrt_m)
        end
        for inv_sqrt_m in inv_sqrt_envs_v2
            Qᵥ₂ = Qᵥ₂ * dag(inv_sqrt_m)
        end

        Aupd = Qᵥ₁ * Rᵥ₁                       # holds the new bond `cind` (in)
        Bupd = Qᵥ₂ * Rᵥ₂                       # holds the new bond `cind` (out)
        cind = only(commoninds(Aupd, Bupd))
        new_cind_dim = dim(cind)

        # Re-split A—B into A —(cind)— ABk —(new_cind)— B with a transparent identity bond
        # vertex: arrows opposite to each neighbour (cind out, new_cind in), parity diagonal.
        bondgr = Rᵥ₁.grading[cind]
        new_cind = Index(new_cind_dim, "Fermion,Link")
        T = scalartype(Aupd)
        Imat = T[i == j ? one(T) : zero(T) for i in 1:new_cind_dim, j in 1:new_cind_dim]
        morder = Index[cind, new_cind]
        mgr = Dictionary{Index, Vector{Bool}}(morder, Vector{Bool}[bondgr, bondgr])
        middle_tensor = FermionicITensor(adapt(dtype)(ITensor(Imat, cind, new_cind)), morder, Bool[false, true], mgr)

        Bupd = replaceinds(Bupd, [cind], [new_cind])
        updated_tensors = FermionicITensor[Aupd, middle_tensor, Bupd]
        if normalize_tensors
            s_values = normalize(s_values)
        end
    end

    if normalize_tensors
        updated_tensors = FermionicITensor[ψᵥ / norm(ψᵥ) for ψᵥ in updated_tensors]
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

        if s_values isa FermionicITensor
            # Re-home the bond spectrum onto each new bond index and install it as the BP
            # message on both orientations (same helper the standard fermionic gate uses).
            b1 = commonind(first(updated_tensors), updated_tensors[2])
            m1, m1r = _bond_spectrum_messages(s_values, b1, e1)
            setmessage!(ψ_bpc, e1, m1)
            setmessage!(ψ_bpc, reverse(e1), m1r)

            b2 = commonind(updated_tensors[2], last(updated_tensors))
            m2, m2r = _bond_spectrum_messages(s_values, b2, e2)
            setmessage!(ψ_bpc, e2, m2)
            setmessage!(ψ_bpc, reverse(e2), m2r)
        else
            ind2 = commonind(s_values, first(updated_tensors))
            s_values = replaceinds(denseblocks(s_values), inds(s_values), [ind2, ind2'])
            setmessage!(ψ_bpc, e1, dag(s_values))
            setmessage!(ψ_bpc, reverse(e1), s_values)

            ind3 = commonind(updated_tensors[2], last(updated_tensors))
            s_values = replaceinds(s_values, [ind2, ind2'], [ind3, ind3'])
            setmessage!(ψ_bpc, e2, dag(s_values))
            setmessage!(ψ_bpc, reverse(e2), s_values)
        end
    end

    return ψ_bpc, err
end

# --- Gate application: pick a bond (1:z) or a site (:A/:B) --------------------

function _gate_tuple(op, verts)
    op isa Tuple || return (op, verts)
    length(op) == 1 ? (op[1], verts) : (op[1], verts, op[2:end]...)
end

_resolve_gate(gate, loc, g, sinds) =
    gate isa ITensor ? gate : first(totensor(_gate_tuple(gate, _obsverts(loc)), g, sinds))

# Cache-aware gate resolution: a pre-built `ITensor`/`FermionicITensor` passes through; a
# circuit-tuple spec is turned into an operator tensor via `tofermionicitensor` (fermionic
# network) or `totensor` (bosonic network).
function _resolve_gate(gate, loc, ψ_bpc::BeliefPropagationCache)
    (gate isa ITensor || gate isa FermionicITensor) && return gate
    net = network(ψ_bpc)
    if is_fermionic(net)
        name = gate isa Tuple ? gate[1] : gate
        params = gate isa Tuple ? gate[2:end] : ()
        s_inds = reduce(vcat, [collect(siteinds(net, v)) for v in _obsverts(loc)])
        return tofermionicitensor(String(name), only(params), s_inds)
    end
    return first(totensor(_gate_tuple(gate, _obsverts(loc)), graph(net), siteinds(net)))
end

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
    O = _resolve_gate(gate, loc, ψ_bpc)
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
