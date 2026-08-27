#=
Backend kernel hooks: the one place where backend-specific fast paths attach to the
generic algorithms. Each generic entry point (`updated_message`, `expect`'s region
contraction, `vertex_scalar`, `simple_update`) calls its hook first; the generic fallback
methods (defined next to those entry points) return `nothing`, so removing this file — or
running networks whose structure the pattern checks reject — leaves the library on the
plain seam-verb path with identical results.

The kernels themselves (buffered contraction chains, fused conjugation, in-place
factorizations) live in the KTensors module; these methods only translate network-level
structure (site indices, environment lists) into kernel inputs.
=#

#Fused fast path for the double-layer BP message update (see KTensors.fused_norm_message).
#Falls through to the generic contraction path when the message structure doesn't match
#(e.g. boundary-MPS messages with link indices).
function norm_message_kernel(tns::TensorNetworkState, v, incoming_ms::Vector{<:KTensor}; normalize)
    ψ = tns[v]
    ψ isa KTensor || return nothing
    sinds = siteinds(tns, v)
    all(i -> i isa KIndex, sinds) || return nothing
    return KTensors.fused_norm_message(ψ, collect(KIndex, sinds), incoming_ms; normalize)
end

#Fused fast path for BP region scalars (expectation-value numerators/denominators and
#vertex scalars): one- and two-vertex regions close through the same fused kernel — a
#two-vertex region is "message from v1 with its operator inserted" followed by a full
#closure at v2. Larger Steiner regions and non-standard structures fall back.
function norm_scalar_kernel(tns::TensorNetworkState, vs::Vector, incoming_ms::Vector{<:KTensor}; op_strings::Function)
    1 <= length(vs) <= 2 || return nothing
    ψs, sindss, ops = KTensor[], Vector{KIndex}[], Union{Nothing, KTensor}[]
    for v in vs
        ψ = tns[v]
        ψ isa KTensor || return nothing
        sinds = siteinds(tns, v)
        all(i -> i isa KIndex, sinds) || return nothing
        str = op_strings(v)
        if str == "I"
            push!(ops, nothing)
        elseif str == "ρ" || length(sinds) != 1
            return nothing
        else
            push!(ops, adapt_like(ψ, op(str, only(sinds))))
        end
        push!(ψs, ψ)
        push!(sindss, collect(KIndex, sinds))
    end

    if length(vs) == 1
        c = KTensors.fused_norm_closure(ψs[1], sindss[1], incoming_ms; op = ops[1])
        (c === nothing || !isempty(inds(c))) && return nothing
        return scalar(c)
    end

    #Partition the region's incoming messages by which vertex tensor they attach to
    ms1, ms2 = KTensor[], KTensor[]
    for m in incoming_ms
        ket_legs = filter(i -> plev(i) == 0, inds(m))
        if all(i -> i ∈ inds(ψs[1]), ket_legs)
            push!(ms1, m)
        elseif all(i -> i ∈ inds(ψs[2]), ket_legs)
            push!(ms2, m)
        else
            return nothing
        end
    end
    T1 = KTensors.fused_norm_closure(ψs[1], sindss[1], ms1; op = ops[1])
    T1 === nothing && return nothing
    c = KTensors.fused_norm_closure(ψs[2], sindss[2], vcat(ms2, [T1]); op = ops[2])
    (c === nothing || !isempty(inds(c))) && return nothing
    return scalar(c)
end

#Fused fast path for the two-site gate (see KTensors.fused_two_site_gate). Falls back on
#unusual apply_kwargs, empty environments, or non-2-index environments.
function fused_simple_update(
        o::KTensor, ψ⃗::Vector{<:KTensor};
        envs, normalize_tensors = true, sqrt_cutoff = nothing, apply_kwargs...
    )
    length(ψ⃗) == 2 || return nothing
    isempty(envs) && return nothing
    all(env -> env isa KTensor && ndims(env) == 2, envs) || return nothing
    isempty(setdiff(keys(apply_kwargs), (:maxdim, :cutoff))) || return nothing

    sqrt_cutoff = isnothing(sqrt_cutoff) ? 10 * eps(real(scalartype(first(envs)))) : sqrt_cutoff
    envs_v1 = filter(env -> hascommoninds(env, ψ⃗[1]), envs)
    envs_v2 = filter(env -> hascommoninds(env, ψ⃗[2]), envs)
    ssi1 = pseudo_sqrt_inv_sqrt.(envs_v1; cutoff = sqrt_cutoff)
    ssi2 = pseudo_sqrt_inv_sqrt.(envs_v2; cutoff = sqrt_cutoff)
    s1 = collect(KIndex, commoninds(ψ⃗[1], o))
    s2 = collect(KIndex, commoninds(ψ⃗[2], o))

    t1, t2, s_values, err = KTensors.fused_two_site_gate(
        o, ψ⃗[1], ψ⃗[2],
        collect(KTensor, first.(ssi1)), collect(KTensor, last.(ssi1)),
        collect(KTensor, first.(ssi2)), collect(KTensor, last.(ssi2)),
        s1, s2; apply_kwargs...
    )
    updated_tensors = [t1, t2]

    if normalize_tensors
        s_values = normalize(s_values)
        for ψᵥ in updated_tensors
            rmul!(data(ψᵥ), inv(norm(ψᵥ)))
        end
    end
    return noprime.(updated_tensors), s_values, err
end

#Direct entry point for circuits already given as backend tensors
function apply_gates(circuit::Vector{<:KTensor}, ψ_bpc::BeliefPropagationCache; kwargs...)
    return _apply_gate_tensors(circuit, ψ_bpc; kwargs...)
end

#Backend tensor gates inside generic (e.g. Any-typed) circuit vectors pass through the
#circuit-tuple path unchanged; the acting vertices are inferred from the site indices.
function toitensor(gate::Union{KTensor, KTensors.TKTensor}, g::NamedGraph, sinds::Dictionary)
    verts = [v for v in keys(sinds) if any(i -> i ∈ inds(gate), sinds[v])]
    return gate, verts
end

# ── TKTensor (graded / fermionic) capability methods ────────────────────────────────────
# Backend-specific counterparts of generic entry points, gathered here with the fused
# dense kernels so the generic files stay backend-agnostic.

#Charged product states on graded (TensorKit-backed) sites: local charges are routed
#through dim-1 links along a spanning tree (a T-join, the recipe validated on the
#fermionic branch) so that every vertex tensor is individually flux-zero — TensorMaps
#enforce zero flux, so a charged site must be neutralized by its links. Summing the
#per-vertex conditions, internal bonds cancel: only the TOTAL charge must vanish.
function graded_tensornetworkstate(eltype, f::Function, g::AbstractGraph, siteinds::Dictionary)
    vs = collect(vertices(g))
    svec = Dictionary(vs, [KTensors.state_vector(f(v), only(siteinds[v])) for v in vs])
    #accumulate subtree charges child → parent over a spanning tree
    acc = Dictionary(vs, [KTensors.vector_sector(svec[v], only(siteinds[v])) for v in vs])
    I = typeof(acc[first(vs)])
    root = first(vs)
    stored = Set(edges(g))
    qedge = Dict{NamedEdge{vertextype(g)}, I}()
    for e in post_order_dfs_edges(g, root)
        c, par = src(e), dst(e)
        #the stored edge carries +q on its src copy, −q on its dst (dual) copy; the
        #subtree below `c` must export its accumulated charge through this bond
        if NamedEdge(c => par) ∈ stored
            qedge[NamedEdge(c => par)] = KTensors.dual_sector(acc[c])
        else
            qedge[NamedEdge(par => c)] = acc[c]
        end
        set!(acc, par, KTensors.fuse_sectors(acc[par], acc[c]))
    end
    #A closed network of flux-zero tensors can only represent a chargeless state (the
    #per-vertex conditions telescope over the bonds). A nonzero TOTAL charge is carried
    #by a dangling dim-1 "Charge"-tagged leg on the root vertex, attached automatically;
    #norm networks pair it bra-ket like an operator-free site leg (see norm_factors).
    triv = KTensors.trivial_sector(acc[root])
    l = Dict(e => KTensors.charged_link_index(get(qedge, e, triv)) for e in edges(g))
    tensors = Dictionary{vertextype(g), Any}()
    for v in vs
        links = KTensors.KIndex[]
        for e in edges(g)
            src(e) == v && push!(links, l[e])
            dst(e) == v && push!(links, dag(l[e]))
        end
        if v == root && acc[root] != triv
            push!(links, KTensors.charged_link_index(KTensors.dual_sector(acc[root]); tags = "Charge"))
        end
        set!(tensors, v, KTensors.product_vertex_tensor(eltype, svec[v], only(siteinds[v]), links))
    end
    tensors = Dictionary(vs, identity.(collect(tensors)))
    #explicit siteinds: a dangling "Charge" leg must not be auto-classified as a site
    return TensorNetworkState(TensorNetwork(tensors, g), siteinds)
end

#Graded boundary-MPS message initialization, following the recipe validated on the
#fermionic branch: a random conserving tensor over centre-biased charged links, instead
#of the rank-1 delta join. The delta join is an exactly symmetric starting point whose
#invariant subspace the (exactly block-preserving) graded fitting cannot leave; a random
#structure-compatible full-rank start converges properly. Conservation itself is free:
#TensorMaps only populate flux-zero trees, so `random_itensor` over correctly-oriented
#legs is the conserving initializer — this function only chooses the link sectors.
function set_graded_interpartition_messages!(bmps_cache::BoundaryMPSCache, es::Vector{<:NamedEdge}; link_sectors = nothing)
    n = length(es)
    #Link i carries the CUMULATIVE charge imbalance of message sites 1..i, so its sector
    #support is the convolution of the per-site charge spectra from the left, intersected
    #(weight-multiplied) with the reachable spectrum from the right. The resulting
    #weights are naturally centre-heavy — for a parity grading this reproduces the
    #even-biased (even ≥ odd) split validated on the fermionic branch.
    spectra = [KTensors.site_charge_spectrum(message(bmps_cache, e)) for e in es]
    prefix = accumulate(KTensors.convolve_charge_spectra, spectra)
    suffix = reverse(accumulate(KTensors.convolve_charge_spectra, reverse(spectra)))
    links = KIndex[]
    for i in 1:(n - 1)
        virt_dim = virtual_index_dimension(bmps_cache, es[i], es[i + 1])
        sp = link_sectors === nothing ?
            KTensors.allocate_link_space(prefix[i], suffix[i + 1], virt_dim) :
            link_sectors(virt_dim)
        push!(links, KIndex(sp, "m$(i)$(i + 1)"))
    end
    for i in 1:n
        m = message(bmps_cache, es[i])
        legs = collect(KIndex, inds(m))
        #left link incoming (non-dual), right link outgoing (dual)
        i > 1 && push!(legs, links[i - 1])
        i < n && push!(legs, dag(links[i]))
        t = adapt_like(m, random_itensor(scalartype(m), legs...))
        iszero(norm(t)) && error(
            "set_graded_interpartition_messages!: no flux-zero blocks on the chosen " *
                "link sectors — the message column carries net charge"
        )
        setmessage!(bmps_cache, es[i], t)
    end
    return bmps_cache
end

#The adjoint of a graded boundary-MPS message in the fitting metric (see
#KTensors.fit_adjoint); the generic fallback in boundarympscache.jl is a plain dag.
function fit_adjoint_message(bmps_cache::BoundaryMPSCache, e::NamedEdge, m::KTensors.TKTensor)
    return KTensors.fit_adjoint(m, _crossing_inds(bmps_cache, e))
end

#Graded purification (infinite-temperature identity) state: per vertex the pairing
#Σₛ |s⟩⟨s| between the ket site legs and their dual-rep ancillas (see
#KTensors.pairing_tensor) — flux-zero per site, so all links are trivial dim-1 with the
#usual src(out)/dst(in) orientation.
function graded_identity_tensornetworkstate(eltype, g::NamedGraph, s::Dictionary)
    ref = first(Iterators.flatten(s))
    l = Dict(e => KTensors.trivial_link_index(ref; tags = "e$(src(e))_$(dst(e))") for e in edges(g))
    ts = Dictionary{vertextype(g), Any}()
    for v in vertices(g)
        ninds = length(s[v])
        ninds % 2 == 0 || error("identity state: odd number of siteinds on vertex $v")
        onehots = [onehot(eltype, (src(e) == v ? l[e] : dag(l[e])) => 1) for e in edges(g) if src(e) == v || dst(e) == v]
        if ninds > 0
            t = KTensors.pairing_tensor(eltype, s[v][1:(ninds ÷ 2)], s[v][((ninds ÷ 2) + 1):ninds])
            set!(ts, v, reduce(*, onehots; init = t))
        else
            set!(ts, v, reduce(*, onehots))
        end
    end
    ts = Dictionary(collect(keys(ts)), identity.(collect(ts)))
    return TensorNetworkState(TensorNetwork(ts, g), s)
end
