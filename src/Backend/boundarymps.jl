struct BoundaryMPSCache{V, PV, BPC<:AbstractBeliefPropagationCache{V, PV},PG} <: AbstractBeliefPropagationCache{V, PV}
    bp_cache::BPC
    partitionedplanargraph::PG
    maximum_virtual_dimension::Int64
end

const _default_boundarymps_update_alg = "orthogonal"
const _default_boundarymps_update_niters = 40
const _default_boundarymps_update_tolerance = 1e-12
const _default_boundarymps_update_cutoff = 1e-12

function default_boundarymps_update_kwargs(; cache_is_flat = false, kwargs...)
    alg = ITensorNetworks.default_message_update_alg(cache_is_flat)
    return (; alg, message_update_kwargs = ITensorNetworks.default_message_update_kwargs(; cache_is_flat, kwargs...))
end

ITensorNetworks.default_message_update_alg(cache_is_flat::Bool = false) = cache_is_flat ? "ITensorMPS" : "orthogonal"

function ITensorNetworks.default_message_update_kwargs(; cache_is_flat = false, cutoff = _default_boundarymps_update_cutoff, kwargs...)
    !cache_is_flat && return return (; niters = _default_boundarymps_update_niters, tolerance = _default_boundarymps_update_tolerance)
    return (; cutoff = cutoff, kwargs...)
end

ITensorNetworks.default_cache_update_kwargs(alg::Algorithm"boundarymps") = default_boundarymps_update_kwargs()

## Frontend functions

"""
    updatecache(bmpsc::BoundaryMPSCache; alg, message_update_kwargs = (; niters, tolerance))

Update the MPS messages inside a boundaryMPS-cache.
"""
function updatecache(bmpsc::BoundaryMPSCache, args...; alg = ITensorNetworks.default_message_update_alg(is_flat(bmpsc)),
    message_update_kwargs = ITensorNetworks.default_message_update_kwargs(; cache_is_flat = is_flat(bmpsc), maxdim = maximum_virtual_dimension(bmpsc)), kwargs...)
    return update(bmpsc, args...; alg, message_update_kwargs, kwargs...)
end

"""
    build_boundarymps_cache(ψ::AbstractITensorNetwork, message_rank::Int64; cache_construction_kwargs = (;), cache_update_kwargs = default_posdef_boundarymps_update_kwargs())

Build the Boundary MPS cache for ψIψ  and update it appropriately
"""
function build_boundarymps_cache(
    ψ::AbstractITensorNetwork,
    message_rank::Int64;
    cache_construction_kwargs = (;),
    cache_update_kwargs = default_boundarymps_update_kwargs(; cache_is_flat = false, maxdim = message_rank),
    update_bp_cache = false,
    update_cache = true
)
    # build the BP cache
    ψIψ = build_bp_cache(ψ; update_cache = update_bp_cache)

    # convert BP cache to boundary MPS cache, no further update needed
    return build_boundarymps_cache(
        ψIψ,
        message_rank;
        cache_construction_kwargs,
        cache_update_kwargs,
        update_cache
    )
end

function build_boundarymps_cache(
    ψIψ::AbstractBeliefPropagationCache,
    message_rank::Int64;
    update_cache = true,
    cache_construction_kwargs = (;),
    cache_update_kwargs = default_boundarymps_update_kwargs(; cache_is_flat = is_flat(ψIψ), maxdim = message_rank),
)

    ψIψ = BoundaryMPSCache(ψIψ; message_rank, cache_construction_kwargs...)

    if update_cache
        ψIψ = updatecache(ψIψ; cache_update_kwargs...)
    end

    return ψIψ
end

is_flat(bmpsc::BoundaryMPSCache) = is_flat(bp_cache(bmpsc))

## Backend functions
bp_cache(bmpsc::BoundaryMPSCache) = bmpsc.bp_cache
partitionedplanargraph(bmpsc::BoundaryMPSCache) = bmpsc.partitionedplanargraph
ppg(bmpsc) = partitionedplanargraph(bmpsc)
maximum_virtual_dimension(bmpsc::BoundaryMPSCache) = bmpsc.maximum_virtual_dimension
planargraph(bmpsc::BoundaryMPSCache) = unpartitioned_graph(partitionedplanargraph(bmpsc))

function ITensorNetworks.partitioned_tensornetwork(bmpsc::BoundaryMPSCache)
    return partitioned_tensornetwork(bp_cache(bmpsc))
end
ITensorNetworks.messages(bmpsc::BoundaryMPSCache) = messages(bp_cache(bmpsc))

function ITensorNetworks.default_bp_maxiter(
    alg::Algorithm,
    bmpsc::BoundaryMPSCache,
)
    return default_bp_maxiter(partitioned_graph(ppg(bmpsc)))
end
function ITensorNetworks.default_edge_sequence(alg::Algorithm, bmpsc::BoundaryMPSCache)
    return pair.(default_edge_sequence(ppg(bmpsc)))
end

default_boundarymps_message_rank(tn::AbstractITensorNetwork) = maxlinkdim(tn)^2
ITensorNetworks.partitions(bmpsc::BoundaryMPSCache) =
    parent.(collect(partitionvertices(ppg(bmpsc))))
NamedGraphs.PartitionedGraphs.partitionedges(bmpsc::BoundaryMPSCache) = pair.(partitionedges(ppg(bmpsc)))

function ITensorNetworks.cache(
    alg::Algorithm"boundarymps",
    tn;
    bp_cache_construction_kwargs = default_cache_construction_kwargs(Algorithm("bp"), tn),
    kwargs...,
)
    return BoundaryMPSCache(
        BeliefPropagationCache(tn; bp_cache_construction_kwargs...);
        kwargs...,
    )
end

function ITensorNetworks.default_cache_construction_kwargs(alg::Algorithm"boundarymps", tn)
    return (;
        bp_cache_construction_kwargs = default_cache_construction_kwargs(
            Algorithm("bp"),
            tn,
        )
    )
end

function Base.copy(bmpsc::BoundaryMPSCache)
    return BoundaryMPSCache(
        copy(bp_cache(bmpsc)),
        copy(ppg(bmpsc)),
        maximum_virtual_dimension(bmpsc),
    )
end

function ITensorNetworks.default_message(
    bmpsc::BoundaryMPSCache,
    pe::PartitionEdge;
    kwargs...,
)
    return default_message(bp_cache(bmpsc), pe::PartitionEdge; kwargs...)
end

#Get the dimension of the virtual index between the two message tensors on pe1 and pe2
function virtual_index_dimension(
    bmpsc::BoundaryMPSCache,
    pe1::PartitionEdge,
    pe2::PartitionEdge,
)
    pes = planargraph_sorted_partitionedges(bmpsc, planargraph_partitionpair(bmpsc, pe1))

    if findfirst(x -> x == pe1, pes) > findfirst(x -> x == pe2, pes)
        lower_pe, upper_pe = pe2, pe1
    else
        lower_pe, upper_pe = pe1, pe2
    end
    inds_above = reduce(vcat, linkinds.((bmpsc,), partitionedges_above(bmpsc, lower_pe)))
    inds_below = reduce(vcat, linkinds.((bmpsc,), partitionedges_below(bmpsc, upper_pe)))

    return Int(minimum((
        prod(Float64.(dim.(inds_above))),
        prod(Float64.(dim.(inds_below))),
        Float64(maximum_virtual_dimension(bmpsc)),
    )))
end

#Vertices of the planargraph
function planargraph_vertices(bmpsc::BoundaryMPSCache, partition)
    return vertices(ppg(bmpsc), PartitionVertex(partition))
end

#Get partition(s) of vertices of the planargraph
function planargraph_partitions(bmpsc::BoundaryMPSCache, vertices::Vector)
    return parent.(partitionvertices(ppg(bmpsc), vertices))
end

function planargraph_partition(bmpsc::BoundaryMPSCache, vertex)
    return only(planargraph_partitions(bmpsc, [vertex]))
end

#Get interpartition pairs of partition edges in the underlying partitioned tensornetwork
function planargraph_partitionpair(bmpsc::BoundaryMPSCache, pe::PartitionEdge)
    return pair(partitionedge(ppg(bmpsc), parent(pe)))
end

#Sort (bottom to top) partitoonedges between pair of partitions in the planargraph
function planargraph_sorted_partitionedges(bmpsc::BoundaryMPSCache, partitionpair::Pair)
    pg = ppg(bmpsc)
    src_vs, dst_vs = vertices(pg, PartitionVertex(first(partitionpair))),
    vertices(pg, PartitionVertex(last(partitionpair)))
    es = reduce(
        vcat,
        [
            [src_v => dst_v for dst_v in intersect(neighbors(pg, src_v), dst_vs)] for
            src_v in src_vs
        ],
    )
    es = sort(NamedEdge.(es); by = x -> findfirst(isequal(src(x)), src_vs))
    return PartitionEdge.(es)
end

#Constructor, inserts missing edge in the planar graph to ensure each partition is connected
#allowing the code to work for arbitrary grids and not just square grids
function BoundaryMPSCache(
    bpc::BeliefPropagationCache;
    grouping_function::Function = v -> first(v),
    group_sorting_function::Function = v -> last(v),
    message_rank::Int64 = default_boundarymps_message_rank(tensornetwork(bpc)),

)
    bpc = insert_pseudo_planar_edges(bpc; grouping_function)
    planar_graph = partitioned_graph(bpc)
    vertex_groups = group(grouping_function, collect(vertices(planar_graph)))
    vertex_groups = map(x -> sort(x; by = group_sorting_function), vertex_groups)
    ppg = PartitionedGraph(planar_graph, vertex_groups)
    bmpsc = BoundaryMPSCache(bpc, ppg, message_rank)
    set_interpartition_messages!(bmpsc)
    return bmpsc
end

function BoundaryMPSCache(tn, args...; kwargs...)
    return BoundaryMPSCache(BeliefPropagationCache(tn, args...); kwargs...)
end

#Functions to get the parellel partitionedges sitting above and below a partitionedge
function partitionedges_above(bmpsc::BoundaryMPSCache, pe::PartitionEdge)
    pes = planargraph_sorted_partitionedges(bmpsc, planargraph_partitionpair(bmpsc, pe))
    pe_pos = only(findall(x -> x == pe, pes))
    return PartitionEdge[pes[i] for i = (pe_pos+1):length(pes)]
end

function partitionedges_below(bmpsc::BoundaryMPSCache, pe::PartitionEdge)
    pes = planargraph_sorted_partitionedges(bmpsc, planargraph_partitionpair(bmpsc, pe))
    pe_pos = only(findall(x -> x == pe, pes))
    return PartitionEdge[pes[i] for i = 1:(pe_pos-1)]
end

function partitionedge_above(bmpsc::BoundaryMPSCache, pe::PartitionEdge)
    pes_above = partitionedges_above(bmpsc, pe)
    isempty(pes_above) && return nothing
    return first(pes_above)
end

function partitionedge_below(bmpsc::BoundaryMPSCache, pe::PartitionEdge)
    pes_below = partitionedges_below(bmpsc, pe)
    isempty(pes_below) && return nothing
    return last(pes_below)
end

#Initialise all the interpartition message tensors
function set_interpartition_messages!(
    bmpsc::BoundaryMPSCache,
    partitionpairs::Vector{<:Pair},
)
    m_keys = keys(messages(bmpsc))
    for partitionpair in partitionpairs
        pes = planargraph_sorted_partitionedges(bmpsc, partitionpair)
        for pe in pes
            if pe ∉ m_keys
                set_message!(bmpsc, pe, ITensor[dense(delta(linkinds(bmpsc, pe)))])
            end
        end
        for i = 1:(length(pes)-1)
            virt_dim = virtual_index_dimension(bmpsc, pes[i], pes[i+1])
            ind = Index(virt_dim, "m$(i)$(i+1)")
            m1, m2 = only(message(bmpsc, pes[i])), only(message(bmpsc, pes[i+1]))
            set_message!(bmpsc, pes[i], ITensor[m1*delta(ind)])
            set_message!(bmpsc, pes[i+1], ITensor[m2*delta(ind)])
        end
    end
    return bmpsc
end

function set_interpartition_messages!(bmpsc::BoundaryMPSCache)
    partitionpairs = pair.(partitionedges(ppg(bmpsc)))
    return set_interpartition_messages!(
        bmpsc,
        vcat(partitionpairs, reverse.(partitionpairs)),
    )
end

#Switch the message tensors on partition edges with their reverse (and dagger them)
function switch_message!(bmpsc::BoundaryMPSCache, pe::PartitionEdge)
    ms = messages(bmpsc)
    me, mer = message(bmpsc, pe), message(bmpsc, reverse(pe))
    set!(ms, pe, dag.(mer))
    set!(ms, reverse(pe), dag.(me))
    return bmpsc
end

function switch_messages!(bmpsc::BoundaryMPSCache, partitionpair::Pair)
    for pe in planargraph_sorted_partitionedges(bmpsc, partitionpair)
        switch_message!(bmpsc, pe)
    end
    return bmpsc
end

function partition_graph(bmpsc::BoundaryMPSCache, partition)
    vs = planargraph_vertices(bmpsc, partition)
    return subgraph(unpartitioned_graph(ppg(bmpsc)), vs)
end

function partition_update!(bmpsc::BoundaryMPSCache, seq::Vector{<:PartitionEdge})
    message_update_function_kwargs = (; normalize = false)
    for pe in seq
        m = updated_message(bp_cache(bmpsc), pe; message_update_function_kwargs)
        set_message!(bmpsc, pe, m)
    end
    return bmpsc
end

#Out-of-place version
function partition_update(bmpsc::BoundaryMPSCache, seq::Vector{<:PartitionEdge})
    bmpsc = copy(bmpsc)
    return partition_update!(bmpsc, seq)
end

#Move the orthogonality centre one step on an interpartition from the message tensor on pe1 to that on pe2
function gauge_step!(
    alg::Algorithm"orthogonal",
    bmpsc::BoundaryMPSCache,
    pe1::PartitionEdge,
    pe2::PartitionEdge;
    kwargs...,
)
    m1, m2 = only(message(bmpsc, pe1)), only(message(bmpsc, pe2))
    @assert !isempty(commoninds(m1, m2))
    left_inds = uniqueinds(m1, m2)
    m1, Y = factorize(m1, left_inds; ortho = "left", kwargs...)
    m2 = m2 * Y
    set_message!(bmpsc, pe1, ITensor[m1])
    set_message!(bmpsc, pe2, ITensor[m2])
    return bmpsc
end

#Move the orthogonality centre via a sequence of steps between message tensors
function gauge_walk!(
    alg::Algorithm,
    bmpsc::BoundaryMPSCache,
    seq::Vector;
    kwargs...,
)
    for (pe1, pe2) in seq
        gauge_step!(alg::Algorithm, bmpsc, pe1, pe2; kwargs...)
    end
    return bmpsc
end

function inserter!(
    alg::Algorithm,
    bmpsc::BoundaryMPSCache,
    update_pe::PartitionEdge,
    m::ITensor;
)
    set_message!(bmpsc, reverse(update_pe), ITensor[dag(m)])
    return bmpsc
end

#Default 1-site extracter
function extracter(
    alg::Algorithm"orthogonal",
    bmpsc::BoundaryMPSCache,
    update_pe::PartitionEdge;
)
    m = only(updated_message(
        bmpsc,
        update_pe;
        message_update_function_kwargs = (; normalize = false)))
    return m
end

function ITensors.commonind(bmpsc::BoundaryMPSCache, pe1::PartitionEdge, pe2::PartitionEdge)
    m1, m2 = message(bmpsc, pe1), message(bmpsc, pe2)
    return commonind(only(m1), only(m2))
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

function ITensorMPS.MPO(bmpsc::BoundaryMPSCache, partition)
    sorted_vs = sort(planargraph_vertices(bmpsc, partition))
    ts = [copy(bmpsc[v]) for v in sorted_vs]
    O = ITensorMPS.MPO(ts)
    return O
end

function ITensorMPS.MPS(bmpsc::BoundaryMPSCache, partitionpair::Pair)
    sorted_pes = planargraph_sorted_partitionedges(bmpsc, partitionpair)
    ms = [only(message(bmpsc, pe)) for pe in sorted_pes]
    return ITensorMPS.MPS(ms)
end

function truncate!(bmpsc::BoundaryMPSCache, partitionpair::Pair; truncate_kwargs...)
    M = ITensorMPS.MPS(bmpsc, partitionpair)
    M = ITensorMPS.truncate(M; truncate_kwargs...)
    return set_interpartition_message!(bmpsc, M, partitionpair)
end

function set_interpartition_message!(bmpsc::BoundaryMPSCache, M::Union{MPS, MPO}, partitionpair::Pair)
    sorted_pes = planargraph_sorted_partitionedges(bmpsc, partitionpair)
    for i in 1:length(M)
        set_message!(bmpsc, sorted_pes[i], ITensor[M[i]])
    end
    return bmpsc
end

function updater!(alg::Algorithm"orthogonal", bmpsc::BoundaryMPSCache, partition_graph, prev_pe, update_pe)
    prev_pe == nothing && return bmpsc

    gauge_step!(alg, bmpsc, reverse(prev_pe), reverse(update_pe))
    pupdate_seq = a_star(partition_graph, parent(src(prev_pe)), parent(src(update_pe)))
    partition_update!(bmpsc, PartitionEdge.(pupdate_seq))
    return bmpsc
end

#Update all the message tensors on an interpartition via an n-site fitting procedure
function ITensorNetworks.update(
    alg::Algorithm"orthogonal",
    bmpsc::BoundaryMPSCache,
    partitionpair::Pair;
    niters::Int64,
    tolerance,
    normalize = true,
)
    bmpsc = copy(bmpsc)
    delete_partition_messages!(bmpsc, first(partitionpair))
    switch_messages!(bmpsc, partitionpair)

    pes = planargraph_sorted_partitionedges(bmpsc, partitionpair)
    pg = partition_graph(bmpsc, first(partitionpair))
    update_seq = vcat([pes[i] for i in 1:length(pes)], [pes[i] for i in (length(pes) - 1):-1:2])

    init_gauge_seq = [(reverse(pes[i]), reverse(pes[i-1])) for i in length(pes):-1:2]
    init_update_seq = post_order_dfs_edges(pg, parent(src(first(update_seq))))
    !isempty(init_gauge_seq) && gauge_walk!(alg, bmpsc, init_gauge_seq)
    !isempty(init_update_seq) && partition_update!(bmpsc, PartitionEdge.(init_update_seq))

    prev_cf, prev_pe = 0, nothing
    for i = 1:niters
        cf = 0
        if i == niters
            update_seq = vcat(update_seq, pes[1])
        end
        for update_pe in update_seq
            updater!(alg, bmpsc, pg, prev_pe, update_pe)
            m = extracter(alg, bmpsc, update_pe)
            n = sqrt((m * dag(m))[])
            cf += n
            if normalize 
                m /= n
            end
            inserter!(alg, bmpsc, update_pe, m)
            prev_pe = update_pe
        end
        epsilon = abs(cf - prev_cf) / length(update_seq)
        !isnothing(tolerance) && epsilon < tolerance && break
        prev_cf = cf
    end
    delete_partition_messages!(bmpsc, first(partitionpair))
    switch_messages!(bmpsc, partitionpair)
    return bmpsc
end

function prev_partitionpair(bmpsc::BoundaryMPSCache, partitionpair::Pair)
    pppg = partitioned_graph(ppg(bmpsc))
    vns = neighbors(pppg, first(partitionpair))
    length(vns) == 1 && return nothing

    @assert length(vns) == 2
    v1, v2 = first(vns), last(vns)
    last(partitionpair) == v1 && return v2 => first(partitionpair)
    last(partitionpair) == v2 && return v1 => first(partitionpair)
end

function generic_apply(O::MPO, M::MPS; kwargs...)
    is_simple_mpo = (length(O) == length(M) && all([length(ITensors.siteinds(O, i)) == 2 for i in 1:length(O)]))
    is_simple_mpo && return ITensorMPS.apply(O, M; kwargs...)

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
    O = ITensorNetwork([i for i in 1:length(O_tensors)], O_tensors)
    O = ITensorNetworks.combine_linkinds(O)
    O = ITensorMPS.MPS([O[v] for v in vertices(O)])
    O = merge_internal_tensors(O)
    return truncate(O; kwargs...)
end

#Update all the message tensors on an interpartition via the ITensorMPS apply function
function ITensorNetworks.update(
    alg::Algorithm"ITensorMPS",
    bmpsc::BoundaryMPSCache,
    partitionpair::Pair;
    cutoff::Number = _default_boundarymps_update_cutoff,
    maxdim::Int = maximum_virtual_dimension(bmpsc),
    kwargs...
)
    bmpsc = copy(bmpsc)
    prev_pp = prev_partitionpair(bmpsc, partitionpair)
    O = ITensorMPS.MPO(bmpsc, first(partitionpair))
    O = ITensorMPS.truncate(O; cutoff, maxdim)
    isnothing(prev_pp) && return set_interpartition_message!(bmpsc, merge_internal_tensors(O), partitionpair)

    M = ITensorMPS.MPS(bmpsc, prev_pp)
    M_out = generic_apply(O, M; cutoff, maxdim)
    return set_interpartition_message!(bmpsc, M_out, partitionpair)
end

#Environment support, assume all vertices live in the same partition for now
function ITensorNetworks.environment(bmpsc::BoundaryMPSCache, verts::Vector; kwargs...)
    vs = parent.((partitionvertices(bp_cache(bmpsc), verts)))
    partition = only(planargraph_partitions(bmpsc, parent.(partitionvertices(bmpsc, verts))))
    pg = partition_graph(bmpsc, partition)
    update_seq = post_order_dfs_edges(pg,first(vs))
    bmpsc = partition_update(bmpsc, PartitionEdge.(update_seq))
    return environment(bp_cache(bmpsc), verts; kwargs...)
end

function ITensorNetworks.region_scalar(bmpsc::BoundaryMPSCache, partition)
    pg = partition_graph(bmpsc, partition)
    v = first(center(pg))
    update_seq = post_order_dfs_edges(pg,v)
    bmpsc = partition_update(bmpsc, PartitionEdge.(update_seq))
    return region_scalar(bp_cache(bmpsc), PartitionVertex(v))
end

function ITensorNetworks.region_scalar(bmpsc::BoundaryMPSCache, verts::Vector)
    partitions = planargraph_partitions(bmpsc, parent.(partitionvertices(bmpsc, verts)))
    if length(partitions) == 1
        return region_scalar(bmpsc, only(partitions))
    end
    error("Contractions involving more than 1 partition not currently supported")
end

function ITensorNetworks.region_scalar(bmpsc::BoundaryMPSCache, partitionpair::Pair)
    pes = planargraph_sorted_partitionedges(bmpsc, partitionpair)
    out = ITensor(one(Bool))
    for pe in pes
        out = (out * (only(message(bmpsc, pe)))) * only(message(bmpsc, reverse(pe)))
    end
    return out[]
end

function add_partitionedges(pg::PartitionedGraph, pes::Vector{<:PartitionEdge})
    g = partitioned_graph(pg)
    g = add_edges(g, parent.(pes))
    return PartitionedGraph(
        unpartitioned_graph(pg),
        g,
        partitioned_vertices(pg),
        which_partition(pg),
    )
end

function add_partitionedges(bpc::BeliefPropagationCache, pes::Vector{<:PartitionEdge})
    pg = add_partitionedges(partitioned_tensornetwork(bpc), pes)
    return BeliefPropagationCache(pg, messages(bpc))
end

#Add partition edges necessary to connect up all vertices in a partition in the planar graph created by the sort function
function insert_pseudo_planar_edges(
    bpc::BeliefPropagationCache;
    grouping_function = v -> first(v),
)
    pg = partitioned_graph(bpc)
    partitions = unique(grouping_function.(collect(vertices(pg))))
    pseudo_edges = PartitionEdge[]
    for p in partitions
        vs = sort(filter(v -> grouping_function(v) == p, collect(vertices(pg))))
        for i = 1:(length(vs)-1)
            if vs[i] ∉ neighbors(pg, vs[i+1])
                push!(pseudo_edges, PartitionEdge(NamedEdge(vs[i] => vs[i+1])))
            end
        end
    end
    return add_partitionedges(bpc, pseudo_edges)
end

pair(pe::PartitionEdge) = parent(src(pe)) => parent(dst(pe))

function delete_partition_messages!(bmpsc::BoundaryMPSCache, partition)
    pg = partition_graph(bmpsc, partition)
    pes = PartitionEdge.(edges(pg))
    pes = vcat(pes, reverse.(pes))
    return delete_messages!(bmpsc, pes)
end

function delete_partitionpair_messages!(bmpsc::BoundaryMPSCache, partitionpair::Pair)
    pes = planargraph_sorted_partitionedges(bmpsc, partitionpair)
    return delete_messages!(bmpsc, pes)
end