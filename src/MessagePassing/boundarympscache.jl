using NamedGraphs.PartitionedGraphs: PartitionedGraph, partitions_graph, partitionvertices, PartitionEdge, partitionedges, partitionedge, PartitionVertex, unpartitioned_graph
using NamedGraphs: add_edges!
using SplitApplyCombine: group

#TODO: Make this show() nicely.
struct BoundaryMPSCache{V, N <: AbstractTensorNetwork{V}, M <: Union{ITensor, Vector{<:ITensor}}} <: AbstractBeliefPropagationCache{V}
    network::N
    messages::Dictionary{NamedEdge, M}
    supergraph::PartitionedGraph
    sorted_edges::Dictionary{PartitionEdge, Vector{NamedEdge}}
    mps_bond_dimension::Integer
end

default_update_alg(bmps_cache::BoundaryMPSCache) = "bp"
function set_default_kwargs(alg::Algorithm"bp", bmps_cache::BoundaryMPSCache)
    maxiter = get(alg.kwargs, :maxiter, default_bp_maxiter(bmps_cache))
    edge_sequence = get(alg.kwargs, :edge_sequence, default_bp_edge_sequence(bmps_cache))
    message_update_alg = set_default_kwargs(
        get(alg.kwargs, :message_update_alg, Algorithm(default_message_update_alg(bmps_cache))), bmps_cache
    )
    return Algorithm("bp"; maxiter, edge_sequence, message_update_alg, tolerance = nothing)
end

function default_bp_edge_sequence(bmps_cache::BoundaryMPSCache)
    return PartitionEdge.(forest_cover_edge_sequence(partitions_graph(supergraph(bmps_cache))))
end
default_bp_maxiter(bmps_cache::BoundaryMPSCache) = is_tree(partitions_graph(supergraph(bmps_cache))) ? 1 : 5
function default_bmps_message_update_alg(tn)
    if tn isa TensorNetworkState || tn isa BilinearForm || tn isa QuadraticForm
        return "orthogonal"
    elseif tn isa TensorNetwork
        return "ITensorMPS"
    end
    return error("Unrecognized network type. Don't know what BMPS message update alg to use.")
end
default_message_update_alg(bmps_cache::BoundaryMPSCache) = default_bmps_message_update_alg(network(bmps_cache))

default_normalize(alg::Algorithm"orthogonal") = true
default_tolerance(bmps_cache::BoundaryMPSCache) = default_tolerance(ITensors.NDTensors.scalartype(bmps_cache))
_default_boundarymps_update_niters = 50
function set_default_kwargs(alg::Algorithm"orthogonal", bmps_cache::BoundaryMPSCache)
    normalize = get(alg.kwargs, :normalize, default_normalize(alg))
    tolerance = get(alg.kwargs, :tolerance, default_tolerance(bmps_cache))
    niters = get(alg.kwargs, :niters, _default_boundarymps_update_niters)
    return Algorithm("orthogonal"; tolerance, niters, normalize)
end

default_normalize(alg::Algorithm"ITensorMPS") = true
function set_default_kwargs(alg::Algorithm"ITensorMPS", bmps_cache::BoundaryMPSCache)
    cutoff = get(alg.kwargs, :cutoff, 1.0e-12)
    normalize = get(alg.kwargs, :normalize, default_normalize(alg))
    return Algorithm("ITensorMPS"; cutoff, normalize)
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

function Base.copy(bmps_cache::BoundaryMPSCache)
    return BoundaryMPSCache(
        copy(network(bmps_cache)),
        copy(messages(bmps_cache)),
        copy(supergraph(bmps_cache)),
        copy(sorted_edges(bmps_cache)),
        mps_bond_dimension(bmps_cache),
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

    inds_above = reduce(vcat, virtualinds.((bmps_cache,), edges_above(bmps_cache, lower_e)))
    inds_below = reduce(vcat, virtualinds.((bmps_cache,), edges_below(bmps_cache, upper_e)))

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
        gauge_state = false
    )
    grouping_function = partition_by == "row" ? v -> first(v) : v -> last(v)
    group_sorting_function = partition_by == "row" ? v -> last(v) : v -> first(v)

    if gauge_state && (tn isa TensorNetworkState)
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
    bmps_cache = BoundaryMPSCache(tn, messages, supergraph, sorted_es, mps_bond_dimension)
    @assert is_correct_format(bmps_cache)
    set_interpartition_messages!(bmps_cache, pes)

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
        for i in 1:(length(es) - 1)
            virt_dim = virtual_index_dimension(bmps_cache, es[i], es[i + 1])
            ind = Index(virt_dim, "m$(i)$(i + 1)")
            m1, m2 = message(bmps_cache, es[i]), message(bmps_cache, es[i + 1])
            t = adapt(datatype(m1))(dense(delta(ind)))
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
    alg = set_default_kwargs(Algorithm("contract", normalize = false, enforce_hermiticity = false), bmps_cache)
    for e in seq
        m = updated_message(alg, bmps_cache, e)
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
        alg::Algorithm"orthogonal",
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
        alg::Algorithm"orthogonal",
        bmps_cache::BoundaryMPSCache,
        update_e::NamedEdge
    )
    message_update_alg = set_default_kwargs(Algorithm("contract"; normalize = false, enforce_hermiticity = false), bmps_cache)
    m = updated_message(message_update_alg, bmps_cache, update_e)
    return m
end

function updater!(alg::Algorithm"orthogonal", bmps_cache::BoundaryMPSCache, partition_graph::AbstractGraph, prev_e, update_e)
    prev_e == nothing && return bmps_cache

    gauge_step!(alg, bmps_cache, reverse(prev_e), reverse(update_e))
    update_seq = a_star(partition_graph, src(prev_e), src(update_e))
    update_partition!(bmps_cache, update_seq)
    return bmps_cache
end

function update_message!(
        alg::Algorithm"orthogonal", bmps_cache::BoundaryMPSCache, pe::PartitionEdge
    )
    delete_partition_messages!(bmps_cache, src(pe))
    switch_messages!(bmps_cache, pe)
    es = sorted_edges(bmps_cache, pe)
    g = partition_graph(bmps_cache, src(pe))
    update_seq = vcat([es[i] for i in 1:length(es)], [es[i] for i in (length(es) - 1):-1:2])

    init_gauge_seq = [(reverse(es[i]), reverse(es[i - 1])) for i in length(es):-1:2]
    init_update_seq = post_order_dfs_edges(g, src(first(update_seq)))
    !isempty(init_gauge_seq) && gauge_walk!(alg, bmps_cache, init_gauge_seq)
    !isempty(init_update_seq) && update_partition!(bmps_cache, init_update_seq)

    prev_cf, prev_e = 0, nothing
    for i in 1:alg.kwargs.niters
        cf = 0
        if i == alg.kwargs.niters
            update_seq = vcat(update_seq, es[1])
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

function merge_internal_tensors(O::Union{MPS, MPO})
    internal_inds = filter(i -> isempty(ITensorMPS.siteinds(O, i)), [i for i in 1:length(O)])

    while !isempty(internal_inds)
        site = first(internal_inds)
        tensors = [O[i] for i in setdiff([i for i in 1:length(O)], [site])]
        if site != length(O)
            tensors[site] = tensors[site] * O[site]
        else
            tensors[site - 1] = tensors[site - 1] * O[site]
        end

        O = typeof(O)(tensors)

        internal_inds = filter(i -> isempty(ITensorMPS.siteinds(O, i)), [i for i in 1:length(O)])
    end
    return O
end

function ITensorMPS.MPO(bmps_cache::BoundaryMPSCache, partition; interpet_as_flat = false)
    @assert network(bmps_cache) isa TensorNetwork || interpet_as_flat
    sorted_vs = sort(vertices(supergraph(bmps_cache), partition))
    ts = [copy(network(bmps_cache)[v]) for v in sorted_vs]
    O = ITensorMPS.MPO(ts)
    return O
end

function ITensorMPS.MPS(bmps_cache::BoundaryMPSCache, pe::PartitionEdge)
    sorted_es = sorted_edges(bmps_cache, pe)
    ms = [message(bmps_cache, e) for e in sorted_es]
    return ITensorMPS.MPS(ms)
end

function truncate!(bmps_cache::BoundaryMPSCache, pe::PartitionEdge; truncate_kwargs...)
    M = ITensorMPS.MPS(bmps_cache, pe)
    M = ITensorMPS.truncate(M; truncate_kwargs...)
    return set_interpartition_message!(bmps_cache, M, pe)
end

function set_interpartition_message!(bmps_cache::BoundaryMPSCache, M::Union{MPS, MPO}, pe::PartitionEdge)
    sorted_es = sorted_edges(bmps_cache, pe)
    for i in 1:length(M)
        setmessage!(bmps_cache, sorted_es[i], M[i])
    end
    return bmps_cache
end

#TODO: Write a generalized zip up fitter. Fix for TensorNetworkState
function generic_apply(O::MPO, M::MPS; normalize = true, kwargs...)
    is_simple_mpo = (length(O) == length(M) && all([length(ITensors.siteinds(O, i)) == 2 for i in 1:length(O)]))

    if is_simple_mpo
        out = ITensorMPS.apply(O, M; alg = "naive", kwargs...)
        if normalize
            out = ITensors.normalize(out)
        end
        return out
    end

    O_tensors = ITensor[]
    for i in 1:length(O)
        m_ind = filter(j -> !isempty(ITensors.commoninds(O[i], M[j])), [j for j in 1:length(M)])
        if isempty(m_ind)
            push!(O_tensors, O[i])
        else
            m_ind = only(m_ind)
            push!(O_tensors, O[i] * M[m_ind])
        end
    end

    #Transform away edges that make a loop
    pairs = reduce(vcat, [[(i, j) for j in (i + 1):length(O_tensors)] for i in 1:length(O_tensors)])
    loop_edges = filter(p -> !isempty(commoninds(O_tensors[first(p)], O_tensors[last(p)])) && abs(first(p) - last(p)) != 1, pairs)
    for (i, j) in loop_edges
        inbetween_vertices = [k for k in (i + 1):(j - 1)]
        edge_to_split = (i, j)
        for k in inbetween_vertices
            cind = only(commoninds(O_tensors[first(edge_to_split)], O_tensors[last(edge_to_split)]))
            d = adapt(datatype(O_tensors[k]))(denseblocks(delta(cind, cind')))
            O_tensors[j] *= d
            O_tensors[k] *= d
            edge_to_split = (k, j)
        end
    end
    for i in 1:(length(O_tensors) - 1)
        cinds = commoninds(O_tensors[i], O_tensors[i + 1])
        if length(cinds) > 1
            combiner = adapt(datatype(O_tensors[i]))(ITensors.combiner(cinds))
            O_tensors[i] *= combiner
            O_tensors[i + 1] *= combiner
        end
    end

    O = ITensorMPS.MPS(O_tensors)
    O = merge_internal_tensors(O)

    if normalize
        O = ITensors.normalize(O)
    end

    return truncate(O; kwargs...)
end

# #Update all the message tensors on an interpartition via the ITensorMPS apply function
function update_message!(
        alg::Algorithm"ITensorMPS",
        bmps_cache::BoundaryMPSCache,
        pe::PartitionEdge;
        maxdim::Integer = mps_bond_dimension(bmps_cache),
    )
    prev_pe = prev_partitionedge(bmps_cache, pe)
    O = ITensorMPS.MPO(bmps_cache, src(pe))
    O = ITensorMPS.truncate(O; alg.kwargs.cutoff, maxdim)
    if isnothing(prev_pe)
        O = merge_internal_tensors(O)
        if alg.kwargs.normalize
            O = ITensors.normalize(O)
        end
        return set_interpartition_message!(bmps_cache, O, pe)
    end

    M = ITensorMPS.MPS(bmps_cache, prev_pe)
    M_out = generic_apply(O, M; cutoff = alg.kwargs.cutoff, normalize = alg.kwargs.normalize, maxdim)
    return set_interpartition_message!(bmps_cache, M_out, pe)
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
    out = ITensor(one(Bool))
    for e in es
        out = (out * (message(bmps_cache, e))) * message(bmps_cache, reverse(e))
    end
    return out[]
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
    partitions = unique(grouping_function.(collect(vertices(g))))
    pseudo_edges = NamedEdge[]
    for p in partitions
        vs = sort(filter(v -> grouping_function(v) == p, collect(vertices(g))))
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
    e_pos = only(findall(x -> x == e, es))
    return NamedEdge[es[i] for i in (e_pos + 1):length(es)]
end

function edges_below(bmps_cache::BoundaryMPSCache, e::NamedEdge)
    es = sorted_edges(bmps_cache, partitionedge(supergraph(bmps_cache), e))
    e_pos = only(findall(x -> x == e, es))
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
