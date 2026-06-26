using NamedGraphs.PartitionedGraphs: PartitionedGraph, partitions_graph, partitionvertices, PartitionEdge, partitionedges, partitionedge, PartitionVertex, unpartitioned_graph
using NamedGraphs: add_edges!
using SplitApplyCombine: group

#TODO: Make this show() nicely.
struct BoundaryMPSCache{V, N <: AbstractTensorNetwork{V}, M <: Union{ITensor, FermionicITensor, Vector{<:ITensor}, Vector{<:FermionicITensor}}} <: AbstractBeliefPropagationCache{V}
    network::N
    messages::Dictionary{NamedEdge, M}
    supergraph::PartitionedGraph
    sorted_edges::Dictionary{PartitionEdge, Vector{NamedEdge}}
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
    return PartitionEdge.(forest_cover_edge_sequence(partitions_graph(supergraph(bmps_cache))))
end
default_bp_maxiter(bmps_cache::BoundaryMPSCache) = is_tree(partitions_graph(supergraph(bmps_cache))) ? 1 : 5
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
default_tolerance(bmps_cache::BoundaryMPSCache) = default_tolerance(ITensors.NDTensors.scalartype(bmps_cache))
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
    effective_graph = partitions_graph(s)
    if !is_ring_graph(effective_graph) && !is_line_graph(effective_graph)
        error("Upon partitioning, graph does not form a line or ring: can't run boundary MPS")
    end
    for pv in partitionvertices(s)
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
function sorted_edges(bmps_cache::BoundaryMPSCache, pe::PartitionEdge)
    return sorted_edges(bmps_cache)[pe]
end

#Forward onto the supergraph
for f in [
        :(NamedGraphs.PartitionedGraphs.partitionvertices),
        :(NamedGraphs.PartitionedGraphs.partitionedges),
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
    es = sorted_edges(bmps_cache, partitionedge(s, e1))

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

    if gauge_state && (tn isa TensorNetworkState) && !is_fermionic(tn)
        tn = gauge_and_scale(tn)
    end
    pseudo_edges = pseudo_planar_edges(tn; grouping_function)
    planar_graph = copy(graph(tn))
    planar_graph = add_edges(planar_graph, pseudo_edges)
    vertex_groups = group(grouping_function, collect(vertices(planar_graph)))
    vertex_groups = map(x -> sort(x; by = group_sorting_function), vertex_groups)
    supergraph = PartitionedGraph(planar_graph, vertex_groups)
    pes = vcat(partitionedges(supergraph), reverse.(partitionedges(supergraph)))
    sorted_es = Dictionary{PartitionEdge, Vector{NamedEdge}}(pes, Vector{NamedEdge}[sorted_edges(supergraph, pe) for pe in pes])

    messages = default_messages()
    bmps_cache = BoundaryMPSCache(tn, messages, supergraph, sorted_es, mps_bond_dimension, Dictionary{Pair, Vector}())
    @assert is_correct_format(bmps_cache)
    set_messages && set_interpartition_messages!(bmps_cache, pes)

    return bmps_cache
end

all_partitionedges(bmps_cache::BoundaryMPSCache) = vcat(partitionedges(bmps_cache), reverse.(partitionedges(bmps_cache)))

#Initialise all the interpartition message tensors
function set_interpartition_messages!(
        bmps_cache::BoundaryMPSCache,
        partitionedges::Vector{<:PartitionEdge} = all_partitionedges(bmps_cache),
    )
    m_keys = keys(messages(bmps_cache))
    for pe in partitionedges
        es = sorted_edges(bmps_cache, pe)
        for e in es
            if e ∉ m_keys
                setmessage!(bmps_cache, e, default_message(bmps_cache, e))
            end
        end
        if !is_fermionic(network(bmps_cache))
            for i in 1:(length(es) - 1)
                virt_dim = virtual_index_dimension(bmps_cache, es[i], es[i + 1])
                ind = Index(virt_dim, "m$(i)$(i + 1)")
                m1, m2 = message(bmps_cache, es[i]), message(bmps_cache, es[i + 1])
                t = adapt_like(m1, dense(delta(ind)))
                setmessage!(bmps_cache, es[i], m1 * t)
                setmessage!(bmps_cache, es[i + 1], m2 * t)
            end
        else
            set_fermionic_interpartition_messages!(bmps_cache, es)
        end
    end
    return bmps_cache
end

function switch_messages!(bmps_cache::BoundaryMPSCache, pe::PartitionEdge)
    for pe in sorted_edges(bmps_cache, pe)
        switch_message!(bmps_cache, pe)
    end
    return bmps_cache
end

function partition_graph(bmps_cache::BoundaryMPSCache, partition::PartitionVertex)
    vs = vertices(supergraph(bmps_cache), partition)
    es = filter(e -> src(e) ∈ vs && dst(e) ∈ vs, edges(supergraph(bmps_cache)))
    g = NamedGraph(vs)
    add_edges!(g, es)
    return g
end

function update_partition!(bmps_cache::BoundaryMPSCache, partition::PartitionVertex)
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
function update_partitions!(bmps_cache::BoundaryMPSCache, partitions::Vector{<:PartitionVertex})
    for p in partitions
        update_partition!(bmps_cache, p)
    end
    return bmps_cache
end

function update_partitions!(bmps_cache::BoundaryMPSCache, vertices::Vector{<:Any})
    partitions = unique(partitionvertices(bmps_cache, vertices))
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
    if !(m1 isa FermionicITensor)
        m1, Y = factorize(m1, left_inds; ortho = "left", kwargs...)
    else
        m1, Y = ITensors.qr(m1, left_inds)
    end
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
        m::Tensor
    )
    # Store the fit-specific adjoint on the reverse slot: it is read back as the bra-rail when
    # building the next site's environment, where its crossing legs close against the bulk via
    # the supertrace (`g·dag`, metric on the crossing legs only — see `fit_adjoint_message`),
    # while its virtual MPS bonds stay plain-`dag` to match the Euclidean-QR canonical form.
    setmessage!(bmps_cache, reverse(update_e), fit_adjoint_message(bmps_cache, update_e, m))
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
        alg::Algorithm"fitting", bmps_cache::BoundaryMPSCache, pe::PartitionEdge
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

function prev_partitionedge(bmps_cache::BoundaryMPSCache, pe::PartitionEdge)
    g = partitions_graph(supergraph(bmps_cache))
    vns = neighbors(g, parent(src(pe)))
    length(vns) == 1 && return nothing
    @assert length(vns) == 2
    v1, v2 = first(vns), last(vns)
    parent(dst(pe)) == v1 && return PartitionEdge(v2 => parent(src(pe)))
    return parent(dst(pe)) == v2 && return PartitionEdge(v1 => parent(src(pe)))
end

function set_interpartition_message!(bmps_cache::BoundaryMPSCache, M::AbstractVector{<:ITensor}, pe::PartitionEdge)
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
        L, R = factorize(T, keep...; ortho = "left", cutoff, maxdim, tags = "Link,l=$i")
        push!(out, L)
        carry = R
        left_link = only(commoninds(L, R))
    end
    @assert !isempty(out) "generic_apply: no outgoing site indices, nothing to build an MPS from"
    carry === nothing || (out[end] *= carry)   # fold leftover norm / trailing internal tensors in

    # Back sweep: right-to-left SVD recompression (optimal truncation of the forward result).
    for i in length(out):-1:2
        bond = only(commoninds(out[i - 1], out[i]))
        L, R = factorize(out[i], bond; ortho = "right", cutoff, maxdim, tags = "Link,l=$(i - 1)")
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
function _bmps_apply_inputs(bmps_cache::BoundaryMPSCache, pe::PartitionEdge; incoming_mps = nothing)
    net = network(bmps_cache)
    sorted_vs = sort(vertices(supergraph(bmps_cache), src(pe)))
    pos = Dict(v => i for (i, v) in enumerate(sorted_vs))
    b = length(sorted_vs)

    # One MPO tensor per site (position) of the source partition.
    mpo = ITensor[net[v] for v in sorted_vs]

    # Incoming MPS, keyed by the site each tensor attaches to.
    mps = Dictionary{Int, ITensor}()
    prev_pe = prev_partitionedge(bmps_cache, pe)
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
        pe::PartitionEdge;
        maxdim::Integer = mps_bond_dimension(bmps_cache),
    )
    mpo, mps, right_inds = _bmps_apply_inputs(bmps_cache, pe)
    out = generic_apply(
        mpo, mps, right_inds;
        cutoff = alg.kwargs.cutoff, maxdim, normalize = alg.kwargs.normalize,
    )
    return set_interpartition_message!(bmps_cache, out, pe)
end

function vertex_scalar(bmps_cache::BoundaryMPSCache, partition::PartitionVertex)
    g = partition_graph(bmps_cache, partition)
    v = first(center(g))
    update_seq = post_order_dfs_edges(g, v)
    bmps_cache = update_partition(bmps_cache, update_seq)
    return vertex_scalar(bmps_cache, v)
end

function edge_scalar(bmps_cache::BoundaryMPSCache, pe::PartitionEdge)
    es = sorted_edges(bmps_cache, pe)
    out = !is_fermionic(network(bmps_cache)) ? ITensor(one(Bool)) : FermionicITensor(adapt(datatype(bmps_cache))(ITensor(1)), Index[], Bool[], Dictionary{Index, Vector{Bool}}())
    for e in es
        out = (out * (message(bmps_cache, e))) * message(bmps_cache, reverse(e))
    end
    return scalar(out)
end

function delete_partition_messages!(bmps_cache::BoundaryMPSCache, partition::PartitionVertex)
    g = partition_graph(bmps_cache, partition)
    es = edges(g)
    es = vcat(es, reverse.(es))
    return deletemessages!(bmps_cache, filter(e -> e ∈ keys(messages(bmps_cache)), es))
end

function delete_interpartition_messages!(bmps_cache::BoundaryMPSCache, pe::PartitionEdge)
    es = sorted_edges(bmps_cache, pe)
    return deletemessages!(bmps_cache, filter(e -> e ∈ keys(messages(bmps_cache)), es))
end

function delete_partition_messages!(bmps_cache::BoundaryMPSCache, partitions::Vector{<:PartitionVertex})
    for p in partitions
        delete_partition_messages!(bmps_cache, p)
    end
    return bmps_cache
end

function delete_partition_messages!(bmps_cache::BoundaryMPSCache, vertices::Vector{<:Any})
    partitions = unique(partitionvertices(bmps_cache, vertices))
    return delete_partition_messages!(bmps_cache, partitions)
end


function vertex_scalars(
        bmps_cache::BoundaryMPSCache, vertices = partitionvertices(supergraph(bmps_cache)); kwargs...
    )
    return map(v -> vertex_scalar(bmps_cache, v; kwargs...), vertices)
end

function edge_scalars(
        bmps_cache::BoundaryMPSCache, edges = partitionedges(bmps_cache); kwargs...
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
    es = sorted_edges(bmps_cache, partitionedge(supergraph(bmps_cache), e))
    e_pos = findfirst(x -> x == e, es)
    return NamedEdge[es[i] for i in (e_pos + 1):length(es)]
end

function edges_below(bmps_cache::BoundaryMPSCache, e::NamedEdge)
    es = sorted_edges(bmps_cache, partitionedge(supergraph(bmps_cache), e))
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
function sorted_edges(pg::PartitionedGraph, pe::PartitionEdge)
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

# Per-vertex factor lists for the boundary-MPS path contraction, keyed by vertex. For a
# bosonic network each vertex's `norm_factors` is independent, so we build them separately.
# For a fermionic network a pair of odd operators (e.g. a hopping `c_i† c_j`) shares a single
# operator-string bond that must be created ONCE across all the path vertices — building the
# factors vertex-by-vertex would make each odd site look parity-forbidden on its own (and
# `norm_factors` would return `nothing`). So the fermionic case uses the grouped builder,
# which threads one shared bond through every vertex. Returns `nothing` if parity-forbidden.
function _path_vertex_factors(net::AbstractTensorNetwork, verts::Vector, op_string_f::Function)
    is_fermionic(net) && return fermionic_norm_factors_grouped(net, verts; op_strings = op_string_f)
    return Dictionary(verts, [norm_factors(net, [v]; op_strings = op_string_f) for v in verts])
end

function path_contract(
        cache::BoundaryMPSCache, vs::Vector{<:Any}, op_string_f::Function; bmps_messages_up_to_date = false,
        calculate_denom = true
    )

    #For boundary MPS, must stay in partition
    partitions = unique(partitionvertices(cache, vs))
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
        # The path visits EVERY vertex of the partition lv1..lv2; build all their factors in one
        # call so a fermionic odd-operator pair shares its single operator-string bond.
        path_vs = vcat([src(e) for e in path], [lv2])
        vertex_factors = _path_vertex_factors(network(cache), path_vs, op_string_f)
        vertex_factors === nothing && return 0, denom
        lv1_vns = neighbors(g, lv1)
        prev_edge = length(lv1_vns) == 1 ? nothing : NamedEdge(setdiff(lv1_vns, [lv2]) => lv1)
        m = length(lv1_vns) == 1 ? nothing : message(cache, prev_edge)
        for e in path
            ignore_edges = prev_edge == nothing ? typeof(e)[reverse(e)] : typeof(e)[reverse(e), prev_edge]
            incoming_ms = incoming_messages(cache, src(e); ignore_edges)
            contract_list = copy(vertex_factors[src(e)])
            append!(contract_list, incoming_ms)
            m != nothing && push!(contract_list, m)

            sequence = contraction_sequence(contract_list; alg = "optimal")
            m = contract(contract_list; sequence)
            prev_edge = e
        end

        contract_list = copy(vertex_factors[lv2])
        incoming_ms = incoming_messages(cache, lv2; ignore_edges = typeof(last(path))[last(path)])
        append!(contract_list, incoming_ms)
        push!(contract_list, m)
        sequence = contraction_sequence(contract_list; alg = "optimal")
        numer = contract(contract_list; sequence)
    else
        vertex_factors = _path_vertex_factors(network(cache), vs, op_string_f)
        vertex_factors === nothing && return 0, denom
        contract_list = copy(vertex_factors[only(vs)])
        incoming_ms = incoming_messages(cache, only(vs))
        append!(contract_list, incoming_ms)
        sequence = contraction_sequence(contract_list; alg = "optimal")
        numer = contract(contract_list; sequence)
    end

    return numer, denom
end

### FERMIONIC HELPERS

# Fermionic interpartition messages: each crossing edge `es[i]` carries the ket/bra
# crossing legs (from `default_message`) plus the MPS virtual bonds shared with its
# neighbours. A virtual bond stitched with a plain rank-1 delta would force the
# message tensor odd whenever the bond carries odd-parity entries, so instead we give
# every virtual bond a graded init and build each message tensor with `random_even_itensor`.
# That guarantees a parity-even start whose virtual bonds can transmit odd-parity flux;
# the orthogonal update then re-derives the true bond gradings via the fermionic QR/SVD.
#
# The even/odd SPLIT must match the parity excess of the double-layer environment the bond
# represents. A double-layer bond from a state bond of dimension χ with `n_e` even / `n_o`
# odd components has `n_e² + n_o²` even and `2 n_e n_o` odd combinations — an EVEN excess of
# `(n_e − n_o)²` whenever χ is odd (and a balanced split when χ is even). Seeding the virtual
# bond with an *odd*-excess grading (e.g. the old alternating `isodd(j)`, which is ⌈D/2⌉ odd)
# under-allocates the even sector for odd χ, and the fit plateaus at a wrong fixed point
# (~1e-4 in the norm). Using `[falses(cld(D,2)); trues(fld(D,2))]` — ⌈D/2⌉ EVEN, the same
# convention the state bonds use (see `random_fermionic_tensornetworkstate`) — gives the
# matching even excess and the orthogonal sweep reproduces the exact partition function for
# both even and odd χ.
_msg_init_grading(D::Integer) = Bool[falses(cld(D, 2)); trues(fld(D, 2))]

function set_fermionic_interpartition_messages!(bmps_cache::BoundaryMPSCache, es::Vector{<:NamedEdge})
    n = length(es)
    # virtual indices + their (even-excess) gradings, shared between consecutive edges
    virtinds = Index[]
    virtgrad = Dictionary{Index, Vector{Bool}}()
    for i in 1:(n - 1)
        virt_dim = virtual_index_dimension(bmps_cache, es[i], es[i + 1])
        ind = Index(virt_dim, "m$(i)$(i + 1)")
        push!(virtinds, ind)
        # even-excess grading (matches the double-layer environment's parity content)
        set!(virtgrad, ind, _msg_init_grading(virt_dim))
    end

    for i in 1:n
        m = message(bmps_cache, es[i])
        legs = copy(m.order)
        dirs = copy(m.dirs)
        gr = copy(m.grading)
        if i > 1                                   # left virtual bond (incoming)
            push!(legs, virtinds[i - 1]); push!(dirs, true)
            set!(gr, virtinds[i - 1], virtgrad[virtinds[i - 1]])
        end
        if i < n                                   # right virtual bond (outgoing)
            push!(legs, virtinds[i]); push!(dirs, false)
            set!(gr, virtinds[i], virtgrad[virtinds[i]])
        end
        T = adapt(datatype(m))(random_even_itensor(scalartype(m), legs, gr))
        setmessage!(bmps_cache, es[i], FermionicITensor(T, legs, dirs, gr))
    end
    return bmps_cache
end

# Crossing legs of an interpartition message: the network's virtual index/indices on
# this edge plus their primes (ket+bra rails). Everything else on a message tensor is a
# virtual MPS bond shared with a neighbour along the partition boundary.
function _crossing_inds(bmps_cache::BoundaryMPSCache, e::NamedEdge)
    cinds = virtualinds(network(bmps_cache), e)
    return Index[cinds; prime.(cinds)]
end

# Fit-specific fermionic adjoint for a boundary-MPS bra-rail tensor: `g·dag` with the
# supertrace metric `g = diag((−1)^p)` applied ONLY on the crossing legs (the network legs the
# bra-rail contracts against the double-layer bulk when the environment is built), NOT on the
# virtual MPS bonds.
#
# Why crossing-only: the crossing legs are where the bra closes against the ket across a
# physical bond, so they need the metric that promotes `contract`'s arrow-driven sign to the
# proper OUT→IN supertrace. The virtual MPS bonds are pure fitting indices orthogonalised by
# the Euclidean LAPACK QR; a metric there would be inconsistent with that QR and break the
# canonical form. With this split the orthogonal sweep reproduces the exact boundary
# environment (hence the exact partition function) for BOTH update directions — applying the
# metric on all legs, or on the bonds only, leaves the two directions inconsistent and the
# alternating iteration converges to a wrong fixed point. Bosonic messages need no metric.
function fit_adjoint_message(bmps_cache::BoundaryMPSCache, e::NamedEdge, m)
    !(m isa FermionicITensor) && return fit_adjoint(m)
    crossing = intersect(m.order, _crossing_inds(bmps_cache, e))
    return fit_adjoint(m, crossing)
end

#Switch the message tensors on partition edges with their reverse (and dagger them)
function switch_message!(bmps_cache::BoundaryMPSCache, e::NamedEdge)
    ms = messages(bmps_cache)
    me, mer = message(bmps_cache, e), message(bmps_cache, reverse(e))
    # `fit_adjoint_message` = `g·dag` (fermions) / `dag` (bosons), with the metric on the
    # crossing legs only (see its definition) so the start/end switch is consistent with the
    # crossing-leg supertrace closure used to build the fit environment.
    set!(ms, e, fit_adjoint_message(bmps_cache, e, mer))
    set!(ms, reverse(e), fit_adjoint_message(bmps_cache, e, me))
    return bmps_cache
end