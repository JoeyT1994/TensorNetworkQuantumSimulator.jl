# TODO: Check for right vertex naming for boundary MPS

efault_expect_update_kwargs() = (; maxiter=50, tol=1e-14)

struct BoundaryMPSBeliefPropagationCache
    bp_cache::BeliefPropagationCache
    pv_partitioning::Dictionary
    group_by_column::Bool
    is_periodic::Bool
end

function BoundaryMPSBeliefPropagationCache(
    ψ::ITensorNetwork; group_by_column=true, max_rank::Int64=1, set_messages=true, scalar_tensornetwork=false
)
    # check that vertices are in the generally correct form
    # we expect `vertices(ψ)` to be `Tuple{Int, Int}` like (1, 2), (2, 3), ...
    if !(keytype(vertices(ψ)) == Tuple{Int64,Int64})
        throw(ArgumentError(
            "Vertices of the cache are not in the correct form. " *
            "Expected 2D coordinates of type Tuple{Int, Int} like (1, 1), (1, 2), .... Got type $(keytype(vertices(ψ)))."
        ))
    end

    if !scalar_tensornetwork
        ψIψ = build_boundarymps_cache(ψ; group_by_column)
    else
        ψIψ = build_flat_boundarymps_cache(ψ; group_by_column)
    end

    partitioned_graphs, is_periodic = construct_partitions(ψIψ; group_by_column, scalar_tensornetwork)
    ψIψ_bpc = BoundaryMPSBeliefPropagationCache(
        ψIψ, partitioned_graphs, group_by_column, is_periodic
    )
    if set_messages
        ψIψ_bpc = initialize_messages(ψIψ_bpc; max_rank)
    end
    return ψIψ_bpc
end


"""
Build the tensor network <ψ|H|ψ> and partition it and define initial messages of dimension r between columns / rows
If H is nothing set it to the identity network
"""
function build_boundarymps_cache(ψ::AbstractITensorNetwork, H=nothing; group_by_column=true)
    ψIψ = H == nothing ? QuadraticFormNetwork(ψ) : QuadraticFormNetwork(H, ψ)
    vertex_groups = if group_by_column
        group(v -> first(first(v)), vertices(ψIψ))
    else
        group(v -> last(first(v)), vertices(ψIψ))
    end
    ψIψ_bpc = BeliefPropagationCache(ψIψ, vertex_groups)
    return ψIψ_bpc
end

"""
Build the tensor network <ψ|H|ψ> and partition it and define initial messages of dimension r between columns / rows
If H is nothing set it to the identity network
"""
function build_flat_boundarymps_cache(ψ::AbstractITensorNetwork; group_by_column=true)
    vertex_groups = if group_by_column
        group(v -> first(v), vertices(ψ))
    else
        group(v -> last(v), vertices(ψ))
    end
    ψ_bpc = BeliefPropagationCache(ψ, vertex_groups)
    return ψ_bpc
end


function construct_partitions(ψIψ::BeliefPropagationCache; group_by_column=true, scalar_tensornetwork=false)
    pvs = partitionvertices(ψIψ)
    pv_partitioning = Dictionary()
    is_periodic = false
    for pv in pvs
        column_state = factor_planar(ψIψ, pv)
        vs_column = collect(vertices(column_state))
        grouping = if group_by_column
            group(v -> !scalar_tensornetwork ? last(first(v)) : last(v), vs_column)
        else
            group(v -> !scalar_tensornetwork ? first(first(v)) : first(v), vs_column)
        end
        pg = PartitionedGraph(column_state, grouping)
        e_periodic = NamedEdge(
            last(sort(vertices(partitioned_graph(pg)))) =>
                first(sort(vertices(partitioned_graph(pg)))),
        )
        if (
            has_edge(partitioned_graph(pg), e_periodic) ||
            has_edge(partitioned_graph(pg), reverse(e_periodic))
        ) && length(vertices(partitioned_graph(pg))) > 2
            is_periodic = true
        end
        pg = merge_internal_tensors(column_state, pg)
        #TODO: Partitioning should actually depend on whether we're connecting left or right, currently this
        # is fixed by calling merge_internal_tensors on the message tensor but I'd rather fix it here in the future
        set!(pv_partitioning, pv, partitioned_vertices(pg))
    end

    return pv_partitioning, is_periodic
end

"""
Get the sub tensor network <ψIψ> for the given partition
"""
function factor_planar(ψIψ_bpc::BeliefPropagationCache, pv::PartitionVertex)
    verts = vertices(ψIψ_bpc, pv)
    return subgraph(tensornetwork(ψIψ_bpc), verts)
end

function merge_internal_tensors(boundary_state::ITensorNetwork, pg::PartitionedGraph)
    external_pvs = filter(
        pv -> !isempty(externalinds(boundary_state, vertices(pg, pv))), partitionvertices(pg)
    )
    while !issetequal(external_pvs, partitionvertices(pg))
        for pv in external_pvs
            pvns = neighbors(pg, pv)
            for pvn in pvns
                if isempty(externalinds(boundary_state, vertices(pg, pvn)))
                    pg = merge_vertices(pg, pv, pvn)
                end
            end
        end
    end
    return pg
end

function merge_internal_tensors(state::ITensorNetwork)
    internal_verts = filter(v -> isempty(siteinds(state, v)), vertices(state))
    while !isempty(internal_verts)
        for v in internal_verts
            vns = neighbors(state, v)
            for vn in vns
                if !isempty(siteinds(state, vn))
                    state = contract(state, NamedEdge(v => vn))
                end
            end
        end
        internal_verts = filter(v -> isempty(siteinds(state, v)), vertices(state))
    end
    return state
end



function externalinds(tn::AbstractITensorNetwork, verts::Vector)
    return reduce(vcat, [collect(uniqueinds(tn, v)) for v in verts])
end

function externalinds(tn::AbstractITensorNetwork)
    return externalinds(tn, collect(vertices(tn)))
end


function initialize_messages(ψIψ::BoundaryMPSBeliefPropagationCache; kwargs...)
    ψIψ = copy(ψIψ)
    ms = messages(ψIψ)
    for (mc, pe) in enumerate(partitionedges(ψIψ))
        set!(ms, pe, default_message(ψIψ, pe; message_counter=mc, kwargs...))
        set!(
            ms, reverse(pe), default_message(ψIψ, reverse(pe); message_counter=mc, kwargs...)
        )
    end
    return ψIψ
end

pv_partitioning(ψIψ::BoundaryMPSBeliefPropagationCache) = ψIψ.pv_partitioning
function pv_partitions(ψIψ::BoundaryMPSBeliefPropagationCache, pv::PartitionVertex)
    return pv_partitioning(ψIψ)[pv]
end
bp_cache(ψIψ::BoundaryMPSBeliefPropagationCache) = ψIψ.bp_cache
group_by_column(ψIψ::BoundaryMPSBeliefPropagationCache) = ψIψ.group_by_column
is_periodic(ψIψ::BoundaryMPSBeliefPropagationCache) = ψIψ.is_periodic

function Base.copy(ψIψ::BoundaryMPSBeliefPropagationCache)
    return BoundaryMPSBeliefPropagationCache(
        copy(bp_cache(ψIψ)), copy(pv_partitioning(ψIψ)), group_by_column(ψIψ), is_periodic(ψIψ)
    )
end


#Forward from BPC:
#Forward from partitioned graph
for f in [
    :(PartitionedGraphs.partitionedge),
    :(PartitionedGraphs.partitionedges),
    :(PartitionedGraphs.partitionvertices),
    :(PartitionedGraphs.partitioned_graph),
    :(PartitionedGraphs.vertices),
    :(PartitionedGraphs.boundary_partitionedges),
    :(ITensorNetworks.linkinds),
    :(ITensorNetworks.messages),
    :(ITensorNetworks.message),
    :(ITensorNetworks.default_edge_sequence),
    :(ITensorNetworks.factors),
    :(ITensorNetworks.default_bp_maxiter),
    :factor_planar,
]
    @eval begin
        function $f(boundarymps_bp_cache::BoundaryMPSBeliefPropagationCache, args...; kwargs...)
            return $f(bp_cache(boundarymps_bp_cache), args...; kwargs...)
        end
    end
end

#Forward from partitioned graph
for f in [:(PartitionedGraphs.partitionedges), :(NamedGraphs.edges)]
    @eval begin
        function $f(bp_cache::BeliefPropagationCache, args...; kwargs...)
            return $f(partitioned_tensornetwork(bp_cache), args...; kwargs...)
        end
    end
end

function default_message(
    ψIψ::BoundaryMPSBeliefPropagationCache,
    pe::PartitionEdge;
    max_rank::Int64=1,
    normalize_message=true,
    message_counter=1,
)
    pv_src, pv_dst = src(pe), dst(pe)
    pv_src_groups, pv_dst_groups = pv_partitions(ψIψ, pv_src), pv_partitions(ψIψ, pv_dst)
    pv_src_state, pv_dst_state = PartitionedGraph(factor_planar(ψIψ, pv_src), pv_src_groups),
    PartitionedGraph(factor_planar(ψIψ, pv_dst), pv_dst_groups)
    site_space = Dictionary(
        parent.(partitionvertices(pv_src_state)),
        [commoninds(pv_src_state, pv_dst_state, pv) for pv in partitionvertices(pv_src_state)],
    )
    s = IndsNetwork(partitioned_graph(pv_src_state); site_space)
    s = insert_edges(s; make_periodic=is_periodic(ψIψ))
    s = if group_by_column(ψIψ)
        rename_vertices(v -> (v, "m" * string(message_counter)), s)
    else
        rename_vertices(v -> (v, "m" * string(message_counter)), s)
    end
    if is_ring(s)
        m = ITensorNetwork(v -> inds -> denseblocks(delta(inds)), s; link_space=max_rank)
    else
        link_space_f = e -> minimum((max_rank, max_bond_dim(s, e)))
        m = ITensorNetwork(v -> inds -> denseblocks(delta(inds)), s, link_space_f)
    end

    m = merge_internal_tensors(m)

    if normalize_message
        m, _ = normalize(m)
    end
    return m
end

"""
Do sequential updates of the MPS messages on the vector `edges`.
Measure the difference (L2 norm) of the new vs old message on each edge
"""
function ITensorNetworks.update(
    ψIψ::BoundaryMPSBeliefPropagationCache,
    edges::Vector{<:PartitionEdge};
    (update_diff!)=nothing,
    kwargs...,
)
    ψIψ_updated = copy(ψIψ)

    mts = messages(ψIψ_updated)
    for e in edges
        set!(mts, e, update_message(ψIψ_updated, e; kwargs...))
        if !isnothing(update_diff!)
            update_diff![] += 1.0
        end
    end
    return ψIψ_updated
end

"""
Generic interface for update, with multiple iterations possible
"""
function ITensorNetworks.update(
    ψIψ::BoundaryMPSBeliefPropagationCache;
    edges=default_edge_sequence(ψIψ),
    maxiter=default_bp_maxiter(ψIψ),
    tol=nothing,
    verbose=false,
    kwargs...,
)
    compute_error = !isnothing(tol)
    if isnothing(maxiter)
        error("You need to specify a number of iterations for BP!")
    end
    for i in 1:maxiter
        diff = compute_error ? Ref(0.0) : nothing
        ψIψ = update(ψIψ, edges; (update_diff!)=diff, kwargs...)
        if compute_error && (diff.x / length(edges)) <= tol
            if verbose
                println("BP converged to desired precision after $i iterations.")
            end
            break
        end
    end
    return ψIψ
end


"""
Update the MPS message in the planar belief propagation cache along the edge pe of the partitioned graph. Use a 1 site DMRG routine
"""
function update_message(
    ψIψ::BoundaryMPSBeliefPropagationCache,
    pe::PartitionEdge;
    density_matrix_alg=true,
    kwargs...,
)
    ϕAψ, ψ = create_message_update_sandwich(ψIψ, pe), dag(message(ψIψ, pe))
    !is_ring(ψ) && return alternating_update_nonperiodic(ψ, ϕAψ; kwargs...)
    !density_matrix_alg && return alternating_update_periodic(ψ, ϕAψ; kwargs...)
    return alternating_update_periodic_densitymatrix(ψ, ϕAψ; kwargs...)
end

function create_message_update_sandwich(ψIψ::BoundaryMPSBeliefPropagationCache, pe::PartitionEdge)
    pe_in = setdiff(boundary_partitionedges(ψIψ, src(pe); dir=:in), [reverse(pe)])
    message_pe = message(ψIψ, pe)
    messages = isempty(pe_in) ? (message_pe,) : (message(ψIψ, only(pe_in)), dag(message_pe))
    return create_sandwich(ψIψ, src(pe), messages)
end


function ITensors.commoninds(
    ptn1::PartitionedGraph, ptn2::PartitionedGraph, pv::PartitionVertex
)
    if pv ∉ partitionvertices(ptn1) || pv ∉ partitionvertices(ptn2)
        return Index{Int64}[]
    end
    ptn1_tensors = ITensor[unpartitioned_graph(ptn1)[v] for v in vertices(ptn1, pv)]
    ptn2_tensors = ITensor[unpartitioned_graph(ptn2)[v] for v in vertices(ptn2, pv)]
    return commoninds(ptn1_tensors, ptn2_tensors)
end

function ITensors.commoninds(A::Vector{ITensor}, B::Vector{ITensor})
    return unique(reduce(vcat, reduce(vcat, [[commoninds(a, b) for a in A] for b in B])))
end

function insert_edges(s::IndsNetwork; make_periodic=false)
    s = copy(s)
    vs = sort(collect(vertices(s)))
    for i in 2:length(vs)
        e = NamedEdge(vs[i-1] => vs[i])
        if e ∉ edges(s) || reverse(e) ∉ edges(s)
            s = add_edge(s, e)
        end
    end

    if make_periodic
        e = NamedEdge(first(vs) => last(vs))
        if e ∉ edges(s) || reverse(e) ∉ edges(s)
            s = add_edge(s, e)
        end
    end
    return s
end

function is_ring(g::AbstractGraph)
    return (is_connected(g) && all([degree(g, v) == 2 for v in vertices(g)]))
end

# from fixes.jl
function ITensorNetwork(
    itensor_constructor::Function, is::IndsNetwork, link_space; kwargs...
)
    is = insert_linkinds(is, edges(is), link_space)
    tn = ITensorNetwork{vertextype(is)}()
    for v in vertices(is)
        add_vertex!(tn, v)
    end
    for e in edges(is)
        add_edge!(tn, e)
    end
    for v in vertices(tn)
        # TODO: Replace with `is[v]` once `getindex(::IndsNetwork, ...)` is smarter.
        siteinds = get(is, v, Index[])
        edges = [edgetype(is)(v, nv) for nv in neighbors(is, v)]
        linkinds = map(e -> is[e], Indices(edges))
        tensor_v = generic_state(itensor_constructor(v), (; siteinds, linkinds))
        setindex_preserve_graph!(tn, tensor_v, v)
    end
    return tn
end

# from fixes.jl
function insert_linkinds(
    indsnetwork::AbstractIndsNetwork, edges, link_space=e -> trivial_space(indsnetwork)
)
    indsnetwork = copy(indsnetwork)
    for e in edges
        # TODO: Change to check if it is empty.
        if !isassigned(indsnetwork, e)
            if !isnothing(link_space)
                iₑ = Index(link_space(e), edge_tag(e))
                # TODO: Allow setting with just `Index`.
                indsnetwork[e] = [iₑ]
            else
                indsnetwork[e] = []
            end
        end
    end
    return indsnetwork
end

function max_bond_dim(s::IndsNetwork, e::NamedEdge; maxrank=10000)
    vsrc_partition, vdst_partition = bipartition(s, e)
    leftinds = reduce(vcat, [s[v] for v in vsrc_partition])
    rightinds = reduce(vcat, [s[v] for v in vdst_partition])
    leftdim = 1
    for ind in leftinds
        leftdim *= dim(ind)
        if leftdim > maxrank
            break
        end
    end
    rightdim = 1
    for ind in rightinds
        rightdim *= dim(ind)
        if rightdim > maxrank
            break
        end
    end
    return minimum((leftdim, rightdim))
end


# generalfunctions.jl 
function bipartition(g::AbstractGraph, e::NamedEdge)
    vsrc, vdst = src(e), dst(e)
    vs = collect(vertices(g))
    vsrc_partition = filter(
        v -> (dist(g, v, vsrc) < dist(g, v, vdst)) && dist(g, v, vsrc) != 0, vs
    )
    vdst_partition = filter(
        v -> (dist(g, v, vsrc) > dist(g, v, vdst)) && dist(g, v, vdst) != 0, vs
    )
    return [vsrc_partition; vsrc], [vdst_partition; vdst]
end

function dist(g, v1, v2)
    return length(a_star(g, v1, v2))
end

function create_sandwich(m::ITensorNetwork, me::ITensorNetwork)
    mm = disjoint_union("ket" => m, "bra" => me)
    return PartitionedGraph(mm, group(v -> first(v), vertices(mm)))
end

function create_sandwich(ψIψ::BoundaryMPSBeliefPropagationCache, pe::PartitionEdge; kwargs...)
    m, me = message(ψIψ, pe), message(ψIψ, reverse(pe))
    return create_sandwich(m, me; kwargs...)
end

function create_sandwich(
    ψIψ::BoundaryMPSBeliefPropagationCache, pvs::Vector{<:PartitionVertex}
)
    messages = [message(ψIψ, pe) for pe in boundary_partitionedges(ψIψ, pvs; dir=:in)]
    return create_sandwich(ψIψ, pvs, Tuple(m for m in messages))
end

function create_sandwich(
    ψIψ::BoundaryMPSBeliefPropagationCache, pvs::Vector{<:PartitionVertex}, messages::Tuple
)
    length(pvs) == 1 && return create_sandwich(ψIψ, only(pvs), messages)
    length(pvs) == 2 && return create_sandwich(ψIψ, first(pvs), last(pvs), messages)
end

function create_sandwich(
    ψ::ITensorNetwork, ψ_partitioning, messages::Tuple; group_by_column=true
)
    sandwich = join_tns(ψ, messages...)
    new_verts = collect(setdiff(vertices(sandwich), vertices(ψ)))
    ψ_partitioning_new = Dictionary()
    for _pv in keys(ψ_partitioning)
        if group_by_column
            set!(
                ψ_partitioning_new,
                _pv,
                vcat(ψ_partitioning[_pv], filter(v -> last(first(v)) == _pv, new_verts)),
            )
        else
            set!(
                ψ_partitioning_new,
                _pv,
                vcat(ψ_partitioning[_pv], filter(v -> first(first(v)) == _pv, new_verts)),
            )
        end
    end
    return PartitionedGraph(sandwich, ψ_partitioning_new)
end

function create_sandwich(
    ψIψ::BoundaryMPSBeliefPropagationCache, pv::PartitionVertex, messages::Tuple
)
    state = factor_planar(ψIψ, pv)
    state_partitioning = copy(pv_partitions(ψIψ, pv))
    return create_sandwich(
        state, state_partitioning, messages; group_by_column=group_by_column(ψIψ)
    )
end

function create_sandwich(
    ψIψ::BoundaryMPSBeliefPropagationCache,
    pv1::PartitionVertex,
    pv2::PartitionVertex,
    messages::Tuple,
)
    state1, state2 = factor_planar(ψIψ, pv1), factor_planar(ψIψ, pv2)
    state = join_tns(state1, state2)
    sandwich = join_tns(state, messages...)
    new_verts = collect(setdiff(vertices(sandwich), vertices(state1)))
    state_partitioning = copy(pv_partitions(ψIψ, pv1))
    for pv in keys(state_partitioning)
        if group_by_column(ψIψ)
            state_partitioning[pv] = vcat(
                state_partitioning[pv], filter(v -> last(first(v)) == pv, new_verts)
            )
        else
            state_partitioning[pv] = vcat(
                state_partitioning[pv], filter(v -> first(first(v)) == pv, new_verts)
            )
        end
    end
    return PartitionedGraph(sandwich, state_partitioning)
end


function join_tns(
    tn1::AbstractITensorNetwork,
    tn2::AbstractITensorNetwork,
    tn_tail::AbstractITensorNetwork...,
)
    return join_tns(join_tns(tn1, tn2), tn_tail...)
end

function join_tns(tn1::AbstractITensorNetwork, tn2::AbstractITensorNetwork)
    return ITensorNetwork(
        vcat([v => tn1[v] for v in vertices(tn1)], [v => tn2[v] for v in vertices(tn2)])
    )
end

### alternatingupdate_nonperiodic.jl
"""
Find the tensor network ψ which maximises <ϕ|A|ψ> / Sqrt(<ψ|ψ>) via a 1-site DMRG routine. 
"""
function alternating_update_nonperiodic(
    ψ::ITensorNetwork,
    ϕAψ::PartitionedGraph;
    niters::Int64=10,
    tolerance=nothing,
    normalize=true,
    verbosity=1,
    kwargs...,
)
    ψ = copy(ψ)
    update_seq = default_update_seq_nonperiodic(ψ)
    ϕAψ_bpc = BeliefPropagationCache(ϕAψ)
    check_convergence = !isnothing(tolerance)

    ϕAψs = zeros(ComplexF64, (niters, length(update_seq)))
    previous_v = nothing
    for i in 1:niters
        for (j, v) in enumerate(update_seq)
            ψ, ϕAψ_bpc, ϕAψs[i, j] = updater(ψ, ϕAψ_bpc, v, previous_v)
            local_state = extracter(ϕAψ_bpc, v; normalize)
            ψ, ϕAψ_bpc = inserter(local_state, ψ, ϕAψ_bpc, v)
            previous_v = v
        end
        if check_convergence &&
           (i > 1) &&
           (abs(mean(ϕAψs[i, :]) - mean(ϕAψs[i-1, :])) / abs(mean(ϕAψs[i, :])) <= tolerance)
            if verbosity == 1
                println("Reached tol")
            end
            return dag(ψ)
        end
    end
    return dag(ψ)
end

function default_update_seq_nonperiodic(
    ψ::AbstractITensorNetwork; nsites::Int=1, filter_path=true
)
    length(vertices(ψ)) == 1 && return [only(vertices(ψ))]
    vs = leaf_vertices(ψ)
    length(vs) == 1 && return vs
    @assert length(vs) == 2
    vstart, vend = first(vs), last(vs)
    path = a_star(ψ, vstart, vend)
    path = vcat(src.(path), dst.(reverse(path)))
    !filter_path && return path
    return filter(v -> !isempty(siteinds(ψ, v)), path)
end


"""
Update ψ by orthogonalizing it onto site v and then contract <ϕ|A|ψ> towards site v with BP, to get exact environments.
Use the knowledge of the previous vertex which was optimised to do this all as efficiently as possible (only need to contract along a path between the new vertex and old one)
"""
function updater(ψ::ITensorNetwork, ϕAψ_bpc::BeliefPropagationCache, v, previous_v)
    if previous_v != nothing
        seq = a_star(ψ, previous_v, v)
        p_seq = a_star(
            partitioned_graph(ϕAψ_bpc),
            parent(only(partitionvertices(ϕAψ_bpc, [previous_v]))),
            parent(only(partitionvertices(ϕAψ_bpc, [v]))),
        )
        if !isnothing(seq)
            ψ = gauge_walk(Algorithm("orthogonalize"), ψ, seq)
        end
        vertices_factors = Dictionary(
            [v for v in vcat(src.(seq), [v])],
            [ψ[v] for v in vcat(src.(seq), [v])],
        )
    else
        seq = v
        p_seq = post_order_dfs_edges(
            partitioned_graph(ϕAψ_bpc),
            parent(only(partitionvertices(ϕAψ_bpc, [v]))),
        )
        if length(vertices(ψ)) > 1
            ψ = tree_orthogonalize(ψ, v)
        end
        vertices_factors =
            Dictionary([v for v in collect(vertices(ψ))], [ψ[v] for v in vertices(ψ)])
    end
    ϕAψ_bpc = update_factors(ϕAψ_bpc, vertices_factors)
    if !isempty(p_seq)
        ϕAψ_bpc = update(
            ϕAψ_bpc,
            PartitionEdge.(p_seq);
            message_update=ms -> default_message_update(ms; normalize=false),
        )
    end
    return ψ, ϕAψ_bpc, region_scalar(ϕAψ_bpc, only(partitionvertices(ϕAψ_bpc, [v])))
end


function extracter(ϕAψ_bpc::BeliefPropagationCache, v; normalize=true)
    local_state = contract(environment(ϕAψ_bpc, [v]); sequence="automatic")
    if normalize
        local_state /= sqrt(norm(local_state))
    end
    return local_state
end

""" 
Set ψ[v] = local_state in both ψ and <ϕ|A|ψ>. Also compute the scalar <ϕ|A|ψ> at the same time
"""
function inserter(
    local_state::ITensor, ψ::ITensorNetwork, ϕAψ_bpc::BeliefPropagationCache, v
)
    ψ = copy(ψ)
    ϕAψ_bpc = update_factor(ϕAψ_bpc, v, dag(local_state))
    ψ[v] = dag(local_state)
    return ψ, ϕAψ_bpc
end

function ITensorNetworks.environment(
    ψIψ::BoundaryMPSBeliefPropagationCache, verts; kwargs...
)
    pvs = partitionvertices(ψIψ, verts)
    ϕAψ = BeliefPropagationCache(create_sandwich(ψIψ, pvs))
    ϕAψ = update(ϕAψ; kwargs...)
    return environment(ϕAψ, verts)
end

function one_site_rdm(ψIψ::BoundaryMPSBeliefPropagationCache, v; kwargs...)
    return contract(environment(ψIψ, [(v, "operator")]; kwargs...); sequence="automatic")
end

#Assume contiguous
function two_site_rdm(ψIψ::BoundaryMPSBeliefPropagationCache, v1, v2; kwargs...)
    return contract(
        environment(ψIψ, [(v1, "operator"), (v2, "operator")]; kwargs...); sequence="automatic"
    )
end