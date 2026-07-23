using NamedGraphs.PartitionedGraphs: PartitionedGraph, quotient_graph, quotientvertices, 
    QuotientEdge, quotientedges, quotientedge, QuotientVertex, unpartitioned_graph, QuotientEdges
using NamedGraphs: add_edges!, NamedDiGraph
using NamedGraphs.GraphsExtensions: directed_graph, undirected_graph, forest_cover_edge_sequence, all_edges
using SplitApplyCombine: group

#TODO: Make this show() nicely.
struct BoundaryMPSCache{V, N <: AbstractTensorNetwork{V}, M <: Union{ITensor, Vector{<:ITensor}}} <: AbstractBeliefPropagationCache{V}
    network::N
    messages::Dictionary{NamedEdge, M}
    supergraph::PartitionedGraph
    sorted_edges::Dictionary{QuotientEdge, Vector{NamedEdge}}
    mps_bond_dimension::Integer
    contraction_sequences::Dictionary{Pair, Vector}
end

default_update_alg(bmps_cache::BoundaryMPSCache) = "bp"
function set_default_kwargs(alg::Algorithm"bp", bmps_cache::BoundaryMPSCache)
    maxiter = get(alg.kwargs, :maxiter, default_bp_maxiter(bmps_cache))
    edge_sequence = get(alg.kwargs, :edge_sequence, bp_edge_sequence(bmps_cache))
    message_update_alg = set_default_kwargs(
        get(alg.kwargs, :message_update_alg, Algorithm(default_message_update_alg(bmps_cache))), bmps_cache
    )
    return Algorithm("bp"; maxiter, edge_sequence, message_update_alg, tolerance = nothing)
end

function bp_edge_sequence(bmps_cache::BoundaryMPSCache)
    return QuotientEdge.(forest_cover_edge_sequence(quotient_graph(supergraph(bmps_cache))))
end
default_bp_maxiter(bmps_cache::BoundaryMPSCache) = is_tree(quotient_graph(supergraph(bmps_cache))) ? 1 : 5
function default_bmps_message_update_alg(tn)
    if tn isa TensorNetworkState || tn isa BilinearForm || tn isa QuadraticForm
        return "fitting"
    elseif tn isa TensorNetwork
        return "zipup"
    end
    return error("Unrecognized network type. Don't know what BMPS message update alg to use.")
end
default_message_update_alg(bmps_cache::BoundaryMPSCache) = default_bmps_message_update_alg(network(bmps_cache))

default_normalize(alg::Algorithm"fitting") = true
default_tolerance(bmps_cache::BoundaryMPSCache) = default_tolerance(scalartype(network(bmps_cache)))
_default_boundarymps_update_niters = 50
function set_default_kwargs(alg::Algorithm"fitting", bmps_cache::BoundaryMPSCache)
    normalize = get(alg.kwargs, :normalize, default_normalize(alg))
    tolerance = get(alg.kwargs, :tolerance, default_tolerance(bmps_cache))
    niters = get(alg.kwargs, :niters, _default_boundarymps_update_niters)
    return Algorithm("fitting"; tolerance, niters, normalize)
end

default_normalize(alg::Algorithm"zipup") = true
function set_default_kwargs(alg::Algorithm"zipup", bmps_cache::BoundaryMPSCache)
    cutoff = get(alg.kwargs, :cutoff, 1.0e-12)
    normalize = get(alg.kwargs, :normalize, default_normalize(alg))
    return Algorithm("zipup"; cutoff, normalize)
end

function default_bmps_update_kwargs(tn::AbstractTensorNetwork)
    verbose = false
    tolerance = nothing
    return (; tolerance, verbose)
end

function default_bmps_update_kwargs(bmps_cache::BoundaryMPSCache)
    maxiter = default_bp_maxiter(bmps_cache)
    return (; default_bmps_update_kwargs(network(bmps_cache))..., maxiter)
end

function is_correct_format(bmps_cache::BoundaryMPSCache)
    s = supergraph(bmps_cache)
    effective_graph = quotient_graph(s)
    if !is_ring_graph(effective_graph) && !is_line_graph(effective_graph)
        error("Upon partitioning, graph does not form a line or ring: can't run boundary MPS")
    end
    for pv in quotientvertices(s)
        if !is_line_graph(subgraph(s, pv))
            error("There's a partition that does not form a line: can't run boundary MPS")
        end
    end
    return true
end

network(bmps_cache::BoundaryMPSCache) = bmps_cache.network
messages(bmps_cache::BoundaryMPSCache) = bmps_cache.messages
supergraph(bmps_cache::BoundaryMPSCache) = bmps_cache.supergraph
graph(bmps_cache::BoundaryMPSCache) = unpartitioned_graph(supergraph(bmps_cache))
mps_bond_dimension(bmps_cache::BoundaryMPSCache) = bmps_cache.mps_bond_dimension
sorted_edges(bmps_cache::BoundaryMPSCache) = bmps_cache.sorted_edges
function sorted_edges(bmps_cache::BoundaryMPSCache, pe::QuotientEdge)
    return sorted_edges(bmps_cache)[pe]
end

#Forward onto the supergraph
for f in [
        :(NamedGraphs.PartitionedGraphs.quotientvertices),
        :(NamedGraphs.PartitionedGraphs.quotientedges),
    ]
    @eval begin
        function $f(bmps_cache::BoundaryMPSCache, args...; kwargs...)
            return $f(supergraph(bmps_cache), args...; kwargs...)
        end
    end
end

contraction_sequences(bmps_cache::BoundaryMPSCache) = bmps_cache.contraction_sequences

function Base.copy(bmps_cache::BoundaryMPSCache)
    return BoundaryMPSCache(
        copy(network(bmps_cache)),
        copy(messages(bmps_cache)),
        copy(supergraph(bmps_cache)),
        copy(sorted_edges(bmps_cache)),
        mps_bond_dimension(bmps_cache),
        copy(contraction_sequences(bmps_cache)),
    )
end

#Get the dimension of the virtual index between the two message tensors on pe1 and pe2
function virtual_index_dimension(
        bmps_cache::BoundaryMPSCache,
        e1::NamedEdge,
        e2::NamedEdge,
    )
    s = supergraph(bmps_cache)
    es = sorted_edges(bmps_cache, quotientedge(s, e1))

    if findfirst(x -> x == e1, es) > findfirst(x -> x == e2, es)
        lower_e, upper_e = e2, e1
    else
        lower_e, upper_e = e1, e2
    end

    inds_above = collect(Iterators.flatten(virtualinds.((bmps_cache,), edges_above(bmps_cache, lower_e))))
    inds_below = collect(Iterators.flatten(virtualinds.((bmps_cache,), edges_below(bmps_cache, upper_e))))

    x1 = prod(Float64.(dim.(inds_above)))
    x2 = prod(Float64.(dim.(inds_below)))
    if network(bmps_cache) isa TensorNetworkState
        return Int(minimum((x1 * x1, x2 * x2, Float64(mps_bond_dimension(bmps_cache)))))
    else
        return Int(minimum((x1, x2, Float64(mps_bond_dimension(bmps_cache)))))
    end
end

function BoundaryMPSCache(
        tn::Union{TensorNetworkState, TensorNetwork, BilinearForm, QuadraticForm},
        mps_bond_dimension::Integer;
        partition_by = "row",
        gauge_state = false,
        set_messages = true,
    )
    grouping_function = partition_by == "row" ? v -> first(v) : v -> last(v)
    group_sorting_function = partition_by == "row" ? v -> last(v) : v -> first(v)

    if gauge_state && (tn isa TensorNetworkState)
        tn = gauge_and_scale(tn)
    end
    pseudo_edges = pseudo_planar_edges(tn; grouping_function)

    planar_graph = add_edges(graph(tn), pseudo_edges)

    vertex_groups = group(grouping_function, collect(vertices(planar_graph)))
    vertex_groups = map(x -> sort(x; by = group_sorting_function), vertex_groups)
    supergraph = PartitionedGraph(planar_graph, vertex_groups)

    pes = all_quotientedges(supergraph)
    sorted_es = Dictionary{QuotientEdge, Vector{NamedEdge}}(pes, Vector{NamedEdge}[sorted_edges(supergraph, pe) for pe in pes])

    messages = default_messages()
    bmps_cache = BoundaryMPSCache(tn, messages, supergraph, sorted_es, mps_bond_dimension, Dictionary{Pair, Vector}())
    @assert is_correct_format(bmps_cache)
    set_messages && set_interpartition_messages!(bmps_cache, pes)

    return bmps_cache
end

all_quotientedges(graph) = QuotientEdges(all_edges(quotient_graph(graph)))

#Initialise all the interpartition message tensors
function set_interpartition_messages!(
        bmps_cache::BoundaryMPSCache,
        quotientedges = all_quotientedges(bmps_cache),
    )
    m_keys = keys(messages(bmps_cache))
    for pe in quotientedges
        es = sorted_edges(bmps_cache, pe)
        for e in es
            if e ∉ m_keys
                setmessage!(bmps_cache, e, default_message(bmps_cache, e))
            end
        end
        for i in 1:(length(es) - 1)
            virt_dim = virtual_index_dimension(bmps_cache, es[i], es[i + 1])
            ind = Index(virt_dim)
            m1, m2 = message(bmps_cache, es[i]), message(bmps_cache, es[i + 1])
            t = adapt_like(m1, dense(delta(ind)))
            setmessage!(bmps_cache, es[i], m1 * t)
            setmessage!(bmps_cache, es[i + 1], m2 * t)
        end
    end
    return bmps_cache
end

#Switch the message tensors on partition edges with their reverse (and dagger them)
function switch_message!(bmps_cache::BoundaryMPSCache, e::NamedEdge)
    ms = messages(bmps_cache)
    me, mer = message(bmps_cache, e), message(bmps_cache, reverse(e))
    set!(ms, e, dag(mer))
    set!(ms, reverse(e), dag(me))
    return bmps_cache
end

function switch_messages!(bmps_cache::BoundaryMPSCache, pe::QuotientEdge)
    for pe in sorted_edges(bmps_cache, pe)
        switch_message!(bmps_cache, pe)
    end
    return bmps_cache
end

function partition_graph(bmps_cache::BoundaryMPSCache, partition::QuotientVertex)
    vs = vertices(supergraph(bmps_cache), partition)
    es = filter(e -> src(e) ∈ vs && dst(e) ∈ vs, edges(supergraph(bmps_cache)))
    g = NamedGraph(vs)
    add_edges!(g, es)
    return g
end

function update_partition!(bmps_cache::BoundaryMPSCache, partition::QuotientVertex)
    g = partition_graph(bmps_cache, partition)
    seq = forest_cover_edge_sequence(g)
    update_partition!(bmps_cache, seq)
    return bmps_cache
end

function update_partition!(bmps_cache::BoundaryMPSCache, seq::Vector)
    isempty(seq) && return bmps_cache
    alg = set_default_kwargs(Algorithm("contract", normalize = false), bmps_cache)
    for e in seq
        m, (cache_key, sequence, seq_changed) = updated_message(alg, bmps_cache, e)
        seq_changed && set!(contraction_sequences(bmps_cache), cache_key, sequence)
        setmessage!(bmps_cache, e, m)
    end
    return bmps_cache
end

function update_partition(bmps_cache::BoundaryMPSCache, args...)
    bmps_cache = copy(bmps_cache)
    return update_partition!(bmps_cache, args...)
end

#Update the messages to be corrected within the given partitions
function update_partitions!(bmps_cache::BoundaryMPSCache, partitions::Vector{<:QuotientVertex})
    for p in partitions
        update_partition!(bmps_cache, p)
    end
    return bmps_cache
end

function update_partitions!(bmps_cache::BoundaryMPSCache, vertices::Vector{<:Any})
    partitions = unique(quotientvertices(bmps_cache, vertices))
    return update_partitions!(bmps_cache, partitions)
end

function update_partitions(bmps_cache::BoundaryMPSCache, args...)
    bmps_cache = copy(bmps_cache)
    return update_partitions!(bmps_cache, args...)
end

# #Move the orthogonality centre one step on an interpartition from the message tensor on pe1 to that on pe2
function gauge_step!(
        alg::Algorithm"fitting",
        bmps_cache::BoundaryMPSCache,
        e1::NamedEdge,
        e2::NamedEdge;
        kwargs...,
    )
    m1, m2 = message(bmps_cache, e1), message(bmps_cache, e2)
    @assert !isempty(commoninds(m1, m2))
    left_inds = uniqueinds(m1, m2)
    m1, Y = factorize(m1, left_inds; ortho = "left", kwargs...)
    m2 = m2 * Y
    setmessage!(bmps_cache, e1, m1)
    setmessage!(bmps_cache, e2, m2)
    return bmps_cache
end

#Move the orthogonality centre via a sequence of steps between message tensors
function gauge_walk!(
        alg::Algorithm,
        bmps_cache::BoundaryMPSCache,
        seq::Vector;
        kwargs...,
    )
    for (e1, e2) in seq
        gauge_step!(alg::Algorithm, bmps_cache, e1, e2; kwargs...)
    end
    return bmps_cache
end

function inserter!(
        alg::Algorithm,
        bmps_cache::BoundaryMPSCache,
        update_e::NamedEdge,
        m::ITensor
    )
    setmessage!(bmps_cache, reverse(update_e), dag(m))
    return bmps_cache
end

#Default 1-site extracter
function extracter(
        alg::Algorithm"fitting",
        bmps_cache::BoundaryMPSCache,
        update_e::NamedEdge
    )
    message_update_alg = set_default_kwargs(Algorithm("contract"; normalize = false), bmps_cache)
    m, _ = updated_message(message_update_alg, bmps_cache, update_e)
    return m
end

function updater!(alg::Algorithm"fitting", bmps_cache::BoundaryMPSCache, partition_graph::AbstractGraph, prev_e, update_e)
    prev_e == nothing && return bmps_cache

    gauge_step!(alg, bmps_cache, reverse(prev_e), reverse(update_e))
    update_seq = a_star(partition_graph, src(prev_e), src(update_e))
    update_partition!(bmps_cache, update_seq)
    return bmps_cache
end

function update_message!(
        alg::Algorithm"fitting", bmps_cache::BoundaryMPSCache, pe::QuotientEdge
    )
    delete_partition_messages!(bmps_cache, src(pe))
    switch_messages!(bmps_cache, pe)
    es = sorted_edges(bmps_cache, pe)
    g = partition_graph(bmps_cache, src(pe))
    update_seq = vcat(es, @view(es[(end - 1):-1:2]))

    init_gauge_seq = [(reverse(es[i]), reverse(es[i - 1])) for i in length(es):-1:2]
    init_update_seq = post_order_dfs_edges(g, src(first(update_seq)))
    !isempty(init_gauge_seq) && gauge_walk!(alg, bmps_cache, init_gauge_seq)
    !isempty(init_update_seq) && update_partition!(bmps_cache, init_update_seq)

    prev_cf, prev_e = 0, nothing
    for i in 1:alg.kwargs.niters
        cf = 0
        if i == alg.kwargs.niters
            push!(update_seq, es[1])
        end
        for update_e in update_seq
            updater!(alg, bmps_cache, g, prev_e, update_e)
            m = extracter(alg, bmps_cache, update_e)
            n = norm(m)
            cf += n
            if alg.kwargs.normalize && n != 0
                m /= n
            end
            inserter!(alg, bmps_cache, update_e, m)
            prev_e = update_e
        end
        cf /= length(update_seq)
        epsilon = abs(cf - prev_cf)
        !isnothing(alg.kwargs.tolerance) && epsilon < alg.kwargs.tolerance && break
        prev_cf = cf
    end
    delete_partition_messages!(bmps_cache, src(pe))
    switch_messages!(bmps_cache, pe)
    return bmps_cache
end

function prev_quotientedge(bmps_cache::BoundaryMPSCache, pe::QuotientEdge)
    g = quotient_graph(supergraph(bmps_cache))
    vns = neighbors(g, parent(src(pe)))
    length(vns) == 1 && return nothing
    @assert length(vns) == 2
    v1, v2 = first(vns), last(vns)
    parent(dst(pe)) == v1 && return QuotientEdge(v2 => parent(src(pe)))
    return parent(dst(pe)) == v2 && return QuotientEdge(v1 => parent(src(pe)))
end

function set_interpartition_message!(bmps_cache::BoundaryMPSCache, M::AbstractVector{<:ITensor}, pe::QuotientEdge)
    sorted_es = sorted_edges(bmps_cache, pe)
    for i in 1:length(M)
        setmessage!(bmps_cache, sorted_es[i], M[i])
    end
    return bmps_cache
end

# Position-indexed MPS·MPO application with truncation: a zip-up forward sweep followed by a
# right-to-left SVD recompression. Works on a raw `Vector{ITensor}` chain so it needs no MPS/MPO
# wrapper types.
#
#   `mpo`        : the contiguous chain of tensors at positions 1:b.
#   `mps`        : incoming MPS tensors keyed by the position of the `mpo` tensor they attach to
#                  (an arbitrary subset of 1:b; a gap is an MPS bond that hops over a site).
#   `right_inds` : per-position outgoing site legs (`right_inds[i]` may be empty); these become the
#                  site indices of the result, one output tensor per non-empty entry.
#
# Returns the truncated result as a `Vector{ITensor}`, one tensor per non-empty `right_inds[i]`, in
# increasing position order.
function generic_apply(
        mpo::Vector{<:ITensor},
        mps::Dictionary{Int, <:ITensor},
        right_inds::Vector{<:Vector{<:Index}};
        cutoff = 0.0,
        maxdim = typemax(Int),
        normalize = true,
    )
    b = length(mpo)
    @assert length(right_inds) == b

    # Forward sweep: carry · MPO[i] · MPS[i], peel off the output legs, truncate the new bond.
    out = ITensor[]
    carry = nothing        # forward environment: singular values + still-open virtual bonds
    left_link = nothing    # bond from the previously emitted output tensor into `carry`
    for i in 1:b
        T = mpo[i]
        haskey(mps, i) && (T *= mps[i])
        carry === nothing || (T = carry * T)

        site = right_inds[i]
        if isempty(site)
            carry = T      # internal / left-only / skipped site: just keep threading the bonds
            continue
        end

        keep = left_link === nothing ? Index[site...] : Index[site..., left_link]
        L, R, _ = factorize_svd(T, keep; ortho = "left", cutoff, maxdim)
        push!(out, L)
        carry = R
        left_link = only(commoninds(L, R))
    end
    @assert !isempty(out) "generic_apply: no outgoing site indices, nothing to build an MPS from"
    carry === nothing || (out[end] *= carry)   # fold leftover norm / trailing internal tensors in

    # Back sweep: right-to-left SVD recompression (optimal truncation of the forward result).
    for i in length(out):-1:2
        bond = only(commoninds(out[i - 1], out[i]))
        L, R, _ = factorize_svd(out[i], [bond]; ortho = "right", cutoff, maxdim)
        out[i] = R
        out[i - 1] *= L
    end

    if normalize
        n = norm(out[1])
        iszero(n) || (out[1] /= n)
    end

    return out
end

# Build the (mpo, mps, right_inds) inputs to `generic_apply` for the outgoing interpartition message
# on `pe`, read directly off the cache geometry (no MPS/MPO wrapper types).
#
# The incoming MPS is sourced either from the cache messages on the previous interpartition
# (`incoming_mps === nothing`, the message-update path) or from a supplied single-layer MPS whose
# tensors are ordered by `sorted_edges(prev_pe)` (the sampling path, where the cache messages are
# doubled ket/bra but only the ket layer is applied). When the source partition is a line endpoint
# there is no previous interpartition, so `mps` comes back empty and the call reduces to compressing
# the MPO chain.
function _bmps_apply_inputs(bmps_cache::BoundaryMPSCache, pe::QuotientEdge; incoming_mps = nothing)
    net = network(bmps_cache)
    sorted_vs = sort(vertices(supergraph(bmps_cache), src(pe)))
    pos = Dict(v => i for (i, v) in enumerate(sorted_vs))
    b = length(sorted_vs)

    # One MPO tensor per site (position) of the source partition.
    mpo = ITensor[net[v] for v in sorted_vs]

    # Incoming MPS, keyed by the site each tensor attaches to.
    mps = Dictionary{Int, ITensor}()
    prev_pe = prev_quotientedge(bmps_cache, pe)
    if prev_pe !== nothing
        for (k, e) in enumerate(sorted_edges(bmps_cache, prev_pe))   # e = prev_v => current_v
            t = incoming_mps === nothing ? message(bmps_cache, e) : incoming_mps[k]
            set!(mps, pos[dst(e)], t)
        end
    end

    # Outgoing site legs: the network index on each edge of `pe`, keyed by the source site.
    right_inds = [Index[] for _ in 1:b]
    for e in sorted_edges(bmps_cache, pe)            # e = current_v => next_v
        right_inds[pos[src(e)]] = collect(virtualinds(net, e))
    end

    return mpo, mps, right_inds
end

# Update all the message tensors on an interpartition via the position-indexed zip-up apply.
function update_message!(
        alg::Algorithm"zipup",
        bmps_cache::BoundaryMPSCache,
        pe::QuotientEdge;
        maxdim::Integer = mps_bond_dimension(bmps_cache),
    )
    mpo, mps, right_inds = _bmps_apply_inputs(bmps_cache, pe)
    out = generic_apply(
        mpo, mps, right_inds;
        cutoff = alg.kwargs.cutoff, maxdim, normalize = alg.kwargs.normalize,
    )
    return set_interpartition_message!(bmps_cache, out, pe)
end

function vertex_scalar(bmps_cache::BoundaryMPSCache, partition::QuotientVertex)
    g = partition_graph(bmps_cache, partition)
    v = first(center(g))
    update_seq = post_order_dfs_edges(g, v)
    bmps_cache = update_partition(bmps_cache, update_seq)
    return vertex_scalar(bmps_cache, v)
end

function edge_scalar(bmps_cache::BoundaryMPSCache, pe::QuotientEdge)
    es = sorted_edges(bmps_cache, pe)
    out = ITensor(one(Bool))
    for e in es
        out = (out * (message(bmps_cache, e))) * message(bmps_cache, reverse(e))
    end
    return scalar(out)
end

function delete_partition_messages!(bmps_cache::BoundaryMPSCache, partition::QuotientVertex)
    g = partition_graph(bmps_cache, partition)
    es = edges(g)
    es = vcat(es, reverse.(es))
    return deletemessages!(bmps_cache, filter(e -> e ∈ keys(messages(bmps_cache)), es))
end

function delete_interpartition_messages!(bmps_cache::BoundaryMPSCache, pe::QuotientEdge)
    es = sorted_edges(bmps_cache, pe)
    return deletemessages!(bmps_cache, filter(e -> e ∈ keys(messages(bmps_cache)), es))
end

function delete_partition_messages!(bmps_cache::BoundaryMPSCache, partitions::Vector{<:QuotientVertex})
    for p in partitions
        delete_partition_messages!(bmps_cache, p)
    end
    return bmps_cache
end

function delete_partition_messages!(bmps_cache::BoundaryMPSCache, vertices::Vector{<:Any})
    partitions = unique(quotientvertices(bmps_cache, vertices))
    return delete_partition_messages!(bmps_cache, partitions)
end


function vertex_scalars(
        bmps_cache::BoundaryMPSCache, vertices = quotientvertices(supergraph(bmps_cache)); kwargs...
    )
    return map(v -> vertex_scalar(bmps_cache, v; kwargs...), vertices)
end

function edge_scalars(
        bmps_cache::BoundaryMPSCache, edges = quotientedges(bmps_cache); kwargs...
    )
    return map(e -> edge_scalar(bmps_cache, e; kwargs...), edges)
end

#PartitionedGraph Helpers
#Add edges necessary to connect up all vertices in a partition in the planar graph created by the sort function
function pseudo_planar_edges(
        g::AbstractGraph;
        grouping_function = v -> first(v),
    )
    all_vs = collect(vertices(g))
    partitions = unique(grouping_function.(all_vs))
    pseudo_edges = NamedEdge[]
    for p in partitions
        vs = sort(filter(v -> grouping_function(v) == p, all_vs))
        for i in 1:(length(vs) - 1)
            if vs[i] ∉ neighbors(g, vs[i + 1])
                push!(pseudo_edges, NamedEdge(vs[i] => vs[i + 1]))
            end
        end
    end
    return pseudo_edges
end

#Functions to get the parellel edges sitting above and below a edge
function edges_above(bmps_cache::BoundaryMPSCache, e::NamedEdge)
    es = sorted_edges(bmps_cache, quotientedge(supergraph(bmps_cache), e))
    e_pos = findfirst(x -> x == e, es)
    return NamedEdge[es[i] for i in (e_pos + 1):length(es)]
end

function edges_below(bmps_cache::BoundaryMPSCache, e::NamedEdge)
    es = sorted_edges(bmps_cache, quotientedge(supergraph(bmps_cache), e))
    e_pos = findfirst(x -> x == e, es)
    return NamedEdge[es[i] for i in 1:(e_pos - 1)]
end

function edge_above(bmps_cache::BoundaryMPSCache, e::NamedEdge)
    es_above = edges_above(bmps_cache, e)
    isempty(es_above) && return nothing
    return first(es_above)
end

function edge_below(bmps_cache::BoundaryMPSCache, e::NamedEdge)
    es_below = edges_below(bmps_cache, e)
    isempty(es_below) && return nothing
    return last(es_below)
end

#Sort (bottom to top) edges between pair of partitions in the planargraph
function sorted_edges(pg::PartitionedGraph, pe::QuotientEdge)
    src_vs, dst_vs = vertices(pg, src(pe)), vertices(pg, dst(pe))
    es = reduce(
        vcat,
        [
            [src_v => dst_v for dst_v in intersect(neighbors(pg, src_v), dst_vs)] for
                src_v in src_vs
        ],
    )
    return sort(NamedEdge.(es); by = x -> findfirst(isequal(src(x)), src_vs))
end

function path_contract(
        cache::BoundaryMPSCache, vs::Vector{<:Any}, op_string_f::Function; bmps_messages_up_to_date = false,
        calculate_denom = true
    )

    #For boundary MPS, must stay in partition
    partitions = unique(quotientvertices(cache, vs))
    length(partitions) > 1 && error("Observable support must be within a single partition (row/ column) of the graph for now.")
    partition = only(partitions)
    g = partition_graph(cache, partition)

    if !bmps_messages_up_to_date
        cache = update_partition(cache, partition)
    end
    denom = calculate_denom ? vertex_scalar(cache, first(vs)) : 0

    if length(vs) > 1
        lvs = leaf_vertices(g)
        @assert length(lvs) == 2
        lv1, lv2 = first(lvs), last(lvs)
        path = a_star(g, lv1, lv2)
        lv1_vns = neighbors(g, lv1)
        prev_edge = length(lv1_vns) == 1 ? nothing : NamedEdge(setdiff(lv1_vns, [lv2]) => lv1)
        m = length(lv1_vns) == 1 ? nothing : message(cache, prev_edge)
        for e in path
            ignore_edges = prev_edge == nothing ? typeof(e)[reverse(e)] : typeof(e)[reverse(e), prev_edge]
            incoming_ms = incoming_messages(cache, src(e); ignore_edges)
            contract_list = norm_factors(network(cache), [src(e)]; op_strings = op_string_f)
            append!(contract_list, incoming_ms)
            m != nothing && push!(contract_list, m)

            sequence = contraction_sequence(contract_list; alg = "optimal")
            m = contract(contract_list; sequence)
            prev_edge = e
        end

        contract_list = norm_factors(network(cache), [lv2]; op_strings = op_string_f)
        incoming_ms = incoming_messages(cache, lv2; ignore_edges = typeof(last(path))[last(path)])
        append!(contract_list, incoming_ms)
        push!(contract_list, m)
        sequence = contraction_sequence(contract_list; alg = "optimal")
        numer = contract(contract_list; sequence)
    else
        contract_list = norm_factors(network(cache), vs; op_strings = op_string_f)
        incoming_ms = incoming_messages(cache, only(vs))
        append!(contract_list, incoming_ms)
        sequence = contraction_sequence(contract_list; alg = "optimal")
        numer = contract(contract_list; sequence)
    end

    return numer, denom
end
