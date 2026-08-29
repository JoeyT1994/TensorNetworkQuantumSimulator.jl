#=
Backend-specific methods, gathered in one file so the generic algorithms stay
backend-agnostic. Two kinds of content:

  * OPTIONAL fast paths: each generic entry point (`updated_message`, `expect`'s region
    contraction, `vertex_scalar`, `simple_update`) calls its hook first, and the generic
    fallbacks return `nothing` — a network whose structure a pattern check rejects takes
    the plain seam-verb path with identical results.
  * REQUIRED graded capability: charged product states (`graded_tensornetworkstate`),
    graded identity/purification constructors, boundary-MPS message initialization over
    charged link spectra, and the tensor-gate passthrough — these are the sole
    implementations of their entry points.

The kernels themselves (buffered contraction chains, fused conjugation, in-place
factorizations) live in the Tensors module; these methods only translate network-level
structure (site indices, environment lists) into kernel inputs.
=#

#Per-vertex operator triage shared by the dense and graded region-scalar kernels:
#"I" → no operator, single-site names → adapted op tensors; ρ insertions and
#multi-site-index vertices fall back to the generic path (returns nothing).
function _region_site_ops(tns::TensorNetworkState, vs::Vector, op_strings::Function, ::Type{T}) where {T}
    ψs, sindss, ops = T[], Vector{Index}[], Union{Nothing, T}[]
    for v in vs
        ψ = tns[v]
        ψ isa T || return nothing
        sinds = siteinds(tns, v)
        str = op_strings(v)
        if str == "I"
            push!(ops, nothing)
        elseif str == "ρ" || length(sinds) != 1
            return nothing
        else
            push!(ops, adapt_like(ψ, op(str, only(sinds))))
        end
        push!(ψs, ψ)
        push!(sindss, collect(Index, sinds))
    end
    return ψs, sindss, ops
end

#Fused fast path for the double-layer BP message update (see Tensors.fused_norm_message).
#Falls through to the generic contraction path when the message structure doesn't match
#(e.g. boundary-MPS messages with link indices).
function norm_message_kernel(tns::TensorNetworkState, v, incoming_ms::Vector{<:Tensor}; normalize)
    ψ = tns[v]
    ψ isa Tensor || return nothing
    return Tensors.fused_norm_message(ψ, collect(Index, siteinds(tns, v)), incoming_ms; normalize)
end

#Fused fast path for BP region scalars (expectation-value numerators/denominators and
#vertex scalars): one- and two-vertex regions close through the same fused kernel — a
#two-vertex region is "message from v1 with its operator inserted" followed by a full
#closure at v2. Larger Steiner regions and non-standard structures fall back.
function norm_scalar_kernel(tns::TensorNetworkState, vs::Vector, incoming_ms::Vector{<:Tensor}; op_strings::Function)
    1 <= length(vs) <= 2 || return nothing
    triage = _region_site_ops(tns, vs, op_strings, Tensor)
    triage === nothing && return nothing
    ψs, sindss, ops = triage

    if length(vs) == 1
        c = Tensors.fused_norm_closure(ψs[1], sindss[1], incoming_ms; op = ops[1])
        (c === nothing || !isempty(inds(c))) && return nothing
        return scalar(c)
    end

    #Partition the region's incoming messages by which vertex tensor they attach to
    ms1, ms2 = Tensor[], Tensor[]
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
    T1 = Tensors.fused_norm_closure(ψs[1], sindss[1], ms1; op = ops[1])
    T1 === nothing && return nothing
    c = Tensors.fused_norm_closure(ψs[2], sindss[2], vcat(ms2, [T1]); op = ops[2])
    (c === nothing || !isempty(inds(c))) && return nothing
    return scalar(c)
end

#Fused fast path for the two-site gate (see Tensors.fused_two_site_gate). Falls back on
#unusual apply_kwargs, empty environments, or non-2-index environments.
function fused_simple_update(
        o::Tensor, ψ⃗::Vector{<:Tensor};
        envs, normalize_tensors = true, sqrt_cutoff = nothing, consume_inputs = false,
        apply_kwargs...
    )
    length(ψ⃗) == 2 || return nothing
    isempty(envs) && return nothing
    all(env -> env isa Tensor && ndims(env) == 2, envs) || return nothing
    isempty(setdiff(keys(apply_kwargs), (:maxdim, :cutoff))) || return nothing
    #all participating tensors must share one supported storage family (host, or one GPU
    #array family) — the kernel's workspace buffer is carved from that same memory
    arrays = Any[o.data]
    append!(arrays, (t.data for t in ψ⃗))
    append!(arrays, (env.data for env in envs))
    Tensors._uniform_kernel_storage(arrays) === nothing && return nothing

    sqrt1, inv1, sqrt2, inv2 = gauged_env_pairs(ψ⃗, envs, sqrt_cutoff)
    s1 = collect(Index, commoninds(ψ⃗[1], o))
    s2 = collect(Index, commoninds(ψ⃗[2], o))

    #storage-consuming path: the outputs are assembled inside the input tensors' own
    #arrays (peak 2(F1+F2) + change instead of 3) — the caller relinquishes the inputs
    t1, t2, s_values, err = Tensors.fused_two_site_gate(
        o, ψ⃗[1], ψ⃗[2],
        collect(Tensor, sqrt1), collect(Tensor, inv1),
        collect(Tensor, sqrt2), collect(Tensor, inv2),
        s1, s2;
        dest1 = consume_inputs ? Tensors._root_storage(ψ⃗[1].data) : nothing,
        dest2 = consume_inputs ? Tensors._root_storage(ψ⃗[2].data) : nothing,
        apply_kwargs...
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
function apply_gates(circuit::Vector{<:Tensor}, ψ_bpc::BeliefPropagationCache; kwargs...)
    return _apply_gate_tensors(circuit, ψ_bpc; kwargs...)
end

#Backend tensor gates inside generic (e.g. Any-typed) circuit vectors pass through the
#circuit-tuple path unchanged; the acting vertices are inferred from the site indices.
function totensor(gate::Union{Tensor, Tensors.GradedTensor}, g::NamedGraph, sinds::Dictionary)
    verts = [v for v in keys(sinds) if any(i -> i ∈ inds(gate), sinds[v])]
    return gate, verts
end

# ── GradedTensor (graded / fermionic) capability methods ────────────────────────────────────
# Backend-specific counterparts of generic entry points, gathered here with the fused
# dense kernels so the generic files stay backend-agnostic.

#Fused graded double-layer closure: absorb messages into the ket sequentially (the
#intermediate stays one-tensor-sized), then close against the bra, materialized once per
#call. `op_tensor === nothing` unprimes the bra sites (a norm closure); otherwise the
#operator bridges the primed bra sites. TensorKit's contraction internals still allocate
#their own permute copies (upstream allocator gap), so the win over the sequence-searched
#generic path is structural only: ~15% fewer allocations, ~20% walltime.
function _graded_closure(ψ::Tensors.GradedTensor, sinds, ms; op_tensor = nothing)
    T = ψ
    for m in ms
        T = T * m
    end
    op_tensor === nothing || (T = T * op_tensor)
    bra = unprime_charge_legs(dag(prime(ψ)), ψ)
    if op_tensor === nothing && !isempty(sinds)
        bra = replaceinds(bra, prime.(sinds), sinds)
    end
    return T * bra
end

#Norm-message structure: every message leg is a ket leg or its prime (boundary-MPS
#messages carry MPS link legs — those fall back to the generic path).
function _is_norm_message(m, ψinds)
    return all(i -> (plev(i) == 0 ? i : noprime(i)) ∈ ψinds, inds(m))
end

function norm_message_kernel(tns::TensorNetworkState, v, incoming_ms::Vector{<:Tensors.GradedTensor}; normalize)
    ψ = tns[v]
    ψ isa Tensors.GradedTensor || return nothing
    all(m -> _is_norm_message(m, inds(ψ)), incoming_ms) || return nothing
    out = _graded_closure(ψ, siteinds(tns, v), incoming_ms)
    if normalize
        n = sum(out)
        iszero(n) || (out = out / n)
    end
    return out
end

function norm_scalar_kernel(tns::TensorNetworkState, vs::Vector, incoming_ms::Vector{<:Tensors.GradedTensor}; op_strings::Function)
    1 <= length(vs) <= 2 || return nothing
    triage = _region_site_ops(tns, vs, op_strings, Tensors.GradedTensor)
    triage === nothing && return nothing
    ψs, sindss, ops = triage
    all(m -> any(ψ -> _is_norm_message(m, inds(ψ)), ψs), incoming_ms) || return nothing

    if length(vs) == 1
        c = _graded_closure(ψs[1], sindss[1], incoming_ms; op_tensor = ops[1])
        return c isa Number ? c : nothing
    end

    #two-vertex region: close v1 with its operator, leaving the shared bond pair open,
    #then feed the result to v2's closure as an extra incoming message
    ms1 = filter(m -> _is_norm_message(m, inds(ψs[1])), incoming_ms)
    ms2 = filter(m -> _is_norm_message(m, inds(ψs[2])), incoming_ms)
    length(ms1) + length(ms2) == length(incoming_ms) || return nothing
    T1 = _graded_closure(ψs[1], sindss[1], ms1; op_tensor = ops[1])
    T1 isa Number && return nothing   #disconnected region: fall back
    c = _graded_closure(ψs[2], sindss[2], vcat(ms2, [T1]); op_tensor = ops[2])
    return c isa Number ? c : nothing
end

#Graded (symmetric, TensorKit-backed) site indices: `sectors` is a list of
#charge => dimension pairs under the group named by `symmetry`. With an even number of
#inds per site (purifications), the second half are ancillas carrying the DUAL
#representation (dag'd copies) so the identity state is flux-zero per site.
function graded_siteinds(sitetype::String, vs::Vector, sitedimension::Integer, sectors, symmetry, inds_per_site::Integer)
    symmetry === nothing && error("siteinds: explicit `sectors` need a `symmetry` name")
    sum(last.(sectors)) == sitedimension ||
        error("siteinds: sector dimensions $(sectors) do not sum to the site dimension $(sitedimension)")
    sp = Tensors.graded_space(symmetry, sectors)
    anc(i) = iseven(inds_per_site) && i > inds_per_site ÷ 2
    return Dictionary(vs, [[(ind = Tensors.Index(sp, site_tag(sitetype)); anc(i) ? dag(ind) : ind) for i in 1:inds_per_site] for v in vs])
end

#Charged product states on graded (TensorKit-backed) sites: local charges are routed
#through dim-1 links along a spanning tree (a T-join, the recipe validated on the
#fermionic branch) so that every vertex tensor is individually flux-zero — TensorMaps
#enforce zero flux, so a charged site must be neutralized by its links. Summing the
#per-vertex conditions, internal bonds cancel: only the TOTAL charge must vanish.
function graded_tensornetworkstate(eltype, f::Function, g::AbstractGraph, siteinds::Dictionary)
    vs = collect(vertices(g))
    svec = Dictionary(vs, [Tensors.state_vector(f(v), only(siteinds[v])) for v in vs])
    #accumulate subtree charges child → parent over a spanning tree
    acc = Dictionary(vs, [Tensors.vector_sector(svec[v], only(siteinds[v])) for v in vs])
    I = typeof(acc[first(vs)])
    root = first(vs)
    stored = Set(edges(g))
    qedge = Dict{NamedEdge{vertextype(g)}, I}()
    for e in post_order_dfs_edges(g, root)
        c, par = src(e), dst(e)
        #the stored edge carries +q on its src copy, −q on its dst (dual) copy; the
        #subtree below `c` must export its accumulated charge through this bond
        if NamedEdge(c => par) ∈ stored
            qedge[NamedEdge(c => par)] = Tensors.dual_sector(acc[c])
        else
            qedge[NamedEdge(par => c)] = acc[c]
        end
        set!(acc, par, Tensors.fuse_sectors(acc[par], acc[c]))
    end
    #A closed network of flux-zero tensors can only represent a chargeless state (the
    #per-vertex conditions telescope over the bonds). A nonzero TOTAL charge is carried
    #by a dangling dim-1 "Charge"-tagged leg on the root vertex, attached automatically;
    #norm networks pair it bra-ket like an operator-free site leg (see norm_factors).
    triv = Tensors.trivial_sector(acc[root])
    l = Dict(e => Tensors.charged_link_index(get(qedge, e, triv)) for e in edges(g))
    tensors = Dictionary{vertextype(g), Any}()
    for v in vs
        links = Tensors.Index[]
        for e in edges(g)
            src(e) == v && push!(links, l[e])
            dst(e) == v && push!(links, dag(l[e]))
        end
        if v == root && acc[root] != triv
            push!(links, Tensors.charged_link_index(Tensors.dual_sector(acc[root]); tags = "Charge"))
        end
        set!(tensors, v, Tensors.product_vertex_tensor(eltype, svec[v], only(siteinds[v]), links))
    end
    tensors = Dictionary(vs, narrow_tensors(collect(tensors)))
    #explicit siteinds: a dangling "Charge" leg must not be auto-classified as a site
    return TensorNetworkState(TensorNetwork(tensors, g), siteinds)
end

#Graded boundary-MPS message initialization, following the recipe validated on the
#fermionic branch: a random conserving tensor over centre-biased charged links, instead
#of the rank-1 delta join. The delta join is an exactly symmetric starting point whose
#invariant subspace the (exactly block-preserving) graded fitting cannot leave; a random
#structure-compatible full-rank start converges properly. Conservation itself is free:
#TensorMaps only populate flux-zero trees, so `random_tensor` over correctly-oriented
#legs is the conserving initializer — this function only chooses the link sectors.
function set_graded_interpartition_messages!(bmps_cache::BoundaryMPSCache, es::Vector{<:NamedEdge})
    n = length(es)
    #Link i carries the CUMULATIVE charge imbalance of message sites 1..i, so its sector
    #support is the convolution of the per-site charge spectra from the left, intersected
    #(weight-multiplied) with the reachable spectrum from the right. The resulting
    #weights are naturally centre-heavy — for a parity grading this reproduces the
    #even-biased (even ≥ odd) split validated on the fermionic branch.
    spectra = [Tensors.site_charge_spectrum(message(bmps_cache, e)) for e in es]
    prefix = accumulate(Tensors.convolve_charge_spectra, spectra)
    suffix = reverse(accumulate(Tensors.convolve_charge_spectra, reverse(spectra)))
    links = Index[]
    for i in 1:(n - 1)
        virt_dim = virtual_index_dimension(bmps_cache, es[i], es[i + 1])
        sp = Tensors.allocate_link_space(prefix[i], suffix[i + 1], virt_dim)
        push!(links, Index(sp, "m$(i)$(i + 1)"))
    end
    for i in 1:n
        m = message(bmps_cache, es[i])
        legs = collect(Index, inds(m))
        #left link incoming (non-dual), right link outgoing (dual)
        i > 1 && push!(legs, links[i - 1])
        i < n && push!(legs, dag(links[i]))
        t = adapt_like(m, random_tensor(scalartype(m), legs...))
        iszero(norm(t)) && error(
            "set_graded_interpartition_messages!: no flux-zero blocks on the chosen " *
                "link sectors — the message column carries net charge"
        )
        setmessage!(bmps_cache, es[i], t)
    end
    return bmps_cache
end

#The adjoint of a graded boundary-MPS message in the fitting metric (see
#Tensors.fit_adjoint); the generic fallback in boundarympscache.jl is a plain dag.
function fit_adjoint_message(bmps_cache::BoundaryMPSCache, e::NamedEdge, m::Tensors.GradedTensor)
    return Tensors.fit_adjoint(m, _crossing_inds(bmps_cache, e))
end

#Graded purification (infinite-temperature identity) state: per vertex the pairing
#Σₛ |s⟩⟨s| between the ket site legs and their dual-rep ancillas (see
#Tensors.pairing_tensor) — flux-zero per site, so all links are trivial dim-1 with the
#usual src(out)/dst(in) orientation.
function graded_identity_tensornetworkstate(eltype, g::NamedGraph, s::Dictionary)
    ref = first(Iterators.flatten(s))
    l = Dict(e => Tensors.trivial_link_index(ref; tags = "e$(src(e))_$(dst(e))") for e in edges(g))
    ts = Dictionary{vertextype(g), Any}()
    for v in vertices(g)
        ninds = length(s[v])
        ninds % 2 == 0 || error("identity state: odd number of siteinds on vertex $v — cannot pair kets with bras")
        onehots = [onehot(eltype, (src(e) == v ? l[e] : dag(l[e])) => 1) for e in edges(g) if src(e) == v || dst(e) == v]
        if ninds > 0
            t = Tensors.pairing_tensor(eltype, s[v][1:(ninds ÷ 2)], s[v][((ninds ÷ 2) + 1):ninds])
            set!(ts, v, reduce(*, onehots; init = t))
        else
            set!(ts, v, reduce(*, onehots))
        end
    end
    ts = Dictionary(collect(keys(ts)), narrow_tensors(collect(ts)))
    return TensorNetworkState(TensorNetwork(ts, g), s)
end
