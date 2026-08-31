#=
Backend-specific capability, gathered in one file so the generic algorithms stay
backend-agnostic: charged product states (`graded_tensornetworkstate`), graded
identity/purification constructors, graded site indices, boundary-MPS message
initialization over charged link spectra, and the tensor-gate passthrough. These are the
sole implementations of their entry points.

One fused fast path lives here too: the double-layer BP closure (message updates and
single-vertex region scalars), which measures 5.1F against the generic path's 6.1F at
parity on walltime, and is the inner loop of every BP sweep. Gate application and
everything else run through the generic seam-verb path — measured, the fused gate was
worse than generic.
=#

#Fused double-layer BP kernel (Tensors.fused_norm_closure): the one specialised path in
#the package, worth it because it is the inner loop of every BP sweep and measures 5.1F
#against the generic path's 6.1F. Anything whose structure it does not recognise — boundary
#MPS messages with MPS link legs, ρ insertions, multi-site-index vertices, >2-vertex
#regions, graded tensors — returns `nothing` and takes the generic path with identical
#results. Gate application is deliberately NOT here: measured, the generic gate is better.
function norm_message_kernel(tns::TensorNetworkState, v, incoming_ms::Vector{<:Tensor}; normalize)
    ψ = tns[v]
    ψ isa Tensor || return nothing
    sinds = siteinds(tns, v)
    all(i -> i isa Index, sinds) || return nothing
    return Tensors.fused_norm_message(ψ, collect(Index, sinds), incoming_ms; normalize)
end

function norm_scalar_kernel(tns::TensorNetworkState, vs::Vector, incoming_ms::Vector{<:Tensor}; op_strings::Function)
    length(vs) == 1 || return nothing
    v = only(vs)
    ψ = tns[v]
    ψ isa Tensor || return nothing
    sinds = siteinds(tns, v)
    all(i -> i isa Index, sinds) || return nothing
    str = op_strings(v)
    o = if str == "I"
        nothing
    elseif str == "ρ" || length(sinds) != 1
        return nothing
    else
        adapt_like(ψ, op(str, only(sinds)))
    end
    c = Tensors.fused_norm_closure(ψ, collect(Index, sinds), incoming_ms; op = o)
    (c === nothing || !isempty(inds(c))) && return nothing
    return scalar(c)
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
