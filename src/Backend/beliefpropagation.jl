const _default_bp_update_maxiter = 25
const _default_bp_update_tol = 1e-7

## Frontend functions

function default_posdef_bp_update_kwargs(; cache_is_tree = false)
    return (; maxiter=default_bp_update_maxiter(cache_is_tree), tol=_default_bp_update_tol, message_update_alg = Algorithm("posdef_contract"))
end

function default_nonposdef_bp_update_kwargs(; cache_is_tree = false)
    return (;  maxiter=default_bp_update_maxiter(cache_is_tree), tol=_default_bp_update_tol, message_update_alg = Algorithm("contract"))
end

function default_square_bp_update_kwargs(; cache_is_tree = false)
    return (;  maxiter=default_bp_update_maxiter(cache_is_tree), tol=_default_bp_update_tol, message_update_alg = Algorithm("squarebp"))
end

function default_bp_update_maxiter(cache_is_tree::Bool = false)
    !cache_is_tree && return _default_bp_update_maxiter
    return 1
end

function ITensorNetworks.updated_message(
    alg::Algorithm"posdef_contract", bpc::AbstractBeliefPropagationCache, edge::PartitionEdge
  )
    updated_messages = updated_message(Algorithm("contract"; alg.kwargs...), bpc, edge)
    return make_hermitian.(updated_messages)
end

ITensorNetworks.default_normalize(alg::Algorithm"posdef_contract") = true
ITensorNetworks.default_sequence_alg(alg::Algorithm"posdef_contract") = "optimal"
function ITensorNetworks.set_default_kwargs(alg::Algorithm"posdef_contract")
    normalize = get(alg.kwargs, :normalize, ITensorNetworks.default_normalize(alg))
    sequence_alg = get(alg.kwargs, :sequence_alg, ITensorNetworks.default_sequence_alg(alg))
    return Algorithm("posdef_contract"; normalize, sequence_alg)
end

"""
    updatecache(bp_cache::BeliefPropagationCache; maxiter::Int64, tol::Number, message_update_kwargs = (; message_update_function = default_message_update))

Update the message tensors inside a bp-cache, running over the graph up to maxiter times until convergence to the desired tolerance `tol`.
"""
function updatecache(bp_cache; maxiter=default_bp_update_maxiter(is_tree(partitions_graph(bp_cache))), tol=_default_bp_update_tol, kwargs...)
    return update(bp_cache; maxiter, tol, kwargs...)
end

"""
    build_normsqr_bp_cache(ψ::ITensorNetwork, args...; kwargs...)

Build the tensornetwork and cache of message tensors for the norm square network `ψIψ`.
"""
function build_normsqr_bp_cache(
    ψ::AbstractITensorNetwork,
    args...;
    update_cache=true,
    cache_update_kwargs=default_posdef_bp_update_kwargs(; cache_is_tree = is_tree(ψ)),
)
    bp_cache = BeliefPropagationCache(QuadraticFormNetwork(ψ), args...)
    # TODO: QuadraticFormNetwork() builds ψIψ network, but for Pauli picture `norm_sqr_network()` is enough
    # https://github.com/ITensor/ITensorNetworks.jl/blob/main/test/test_belief_propagation.jl line 49 to construct the cache without the identities.
    if update_cache
        bp_cache = updatecache(bp_cache; cache_update_kwargs...)
    end
    return bp_cache
end

"""
    is_flat(bpc::BeliefPropagationCache)

Is the network inside bpc `flat', i.e. does every partition contain only one tensor
"""
function is_flat(bpc::BeliefPropagationCache)
    pg = partitioned_tensornetwork(bpc)
    return all([length(vertices(pg, pv)) == 1 for pv in partitionvertices(pg)])
end

"""
    symmetric_gauge(ψ::AbstractITensorNetwork; cache_update_kwargs = default_posdef_bp_update_kwargs(), kwargs...)

Transform a tensor netework into the symmetric gauge, where the BP message tensors are all diagonal
"""
function symmetric_gauge(ψ::AbstractITensorNetwork; cache_update_kwargs=default_posdef_bp_update_kwargs(; cache_is_tree = is_tree(ψ)), kwargs...)
    ψ_vidal = VidalITensorNetwork(ψ; cache_update_kwargs, kwargs...)
    cache_ref = Ref{BeliefPropagationCache}()
    ψ_symm = ITensorNetwork(ψ_vidal; (cache!)=cache_ref)
    bp_cache = cache_ref[]
    return ψ_symm, bp_cache
end

"""
    LinearAlgebra.normalize(ψ::ITensorNetwork, ψψ_bpc::BeliefPropagationCache; cache_update_kwargs = default_posdef_bp_update_kwargs(), update_cache = false)

Scale a tensor netework and its norm_sqr cache such that ψIψ = 1 under the BP approximation
"""
function LinearAlgebra.normalize(
    ψ::ITensorNetwork,
    ψψ_bpc::BeliefPropagationCache;
    cache_update_kwargs=default_posdef_bp_update_kwargs(; cache_is_tree = is_tree(ψ)),
    update_cache=false,
)
    ψψ_bpc_ref = Ref(copy(ψψ_bpc))
    ψ = normalize(ψ; alg="bp", (cache!)=ψψ_bpc_ref, cache_update_kwargs, update_cache)

    return ψ, ψψ_bpc_ref[]
end

"""
    ITensors.scalar(bp_cache::AbstractBeliefPropagationCache; alg = "bp", cache_update_kwargs)

Compute the contraction of the tensor network inside the bp_cache with different algorithm choices
"""
function ITensors.scalar(
    bp_cache::AbstractBeliefPropagationCache,
    args...;
    alg="bp",
    kwargs...,
)
    return scalar(Algorithm(alg), bp_cache, args...; kwargs...)
end

function ITensors.scalar(alg::Algorithm"bp", bp_cache::AbstractBeliefPropagationCache)
    return scalar(bp_cache)
end

"""
    ITensorNetworks.region_scalar(bpc::BeliefPropagationCache, verts::Vector)

Compute contraction involving incoming messages to the contiguous set of tensors on the given vertices
"""
function ITensorNetworks.region_scalar(bpc::BeliefPropagationCache, verts::Vector)
    partitions = partitionvertices(bpc, verts)
    length(partitions) == 1 && return region_scalar(bpc, only(partitions))
    if length(partitions) == 2
        p1, p2 = first(partitions), last(partitions)
        if parent(p1) ∉ neighbors(partitions_graph(bpc), parent(p2))
            error(
                "Only contractions involving neighboring partitions are currently supported",
            )
        end
        ms = incoming_messages(bpc, partitions)
        local_tensors = factors(bpc, partitions)
        ts = [ms; local_tensors]
        seq = contraction_sequence(ts; alg="optimal")
        return contract(ts; sequence=seq)[]
    end
    error("Contractions involving more than 2 partitions not currently supported")
    return nothing
end


"""
    entanglement(ψ::ITensorNetwork, e::NamedEdge; (cache!) = nothing, cache_update_kwargs = default_posdef_bp_update_kwargs())

Bipartite Von-Neumann entanglement entropy, estimated, via BP, using the spectrum of the bond tensor on the given edge.
"""
function entanglement(
    ψ::ITensorNetwork,
    e::NamedEdge;
    (cache!)=nothing,
    cache_update_kwargs=default_posdef_bp_update_kwargs(; cache_is_tree = is_tree(ψ)),
)
    cache = isnothing(cache!) ? build_normsqr_bp_cache(ψ; cache_update_kwargs) : cache![]
    ψ_vidal = VidalITensorNetwork(ψ; cache)
    bt = ITensorNetworks.bond_tensor(ψ_vidal, e)
    ee = 0
    for d in diag(bt)
        ee -= abs(d) >= eps(eltype(bt)) ? d * d * log2(d * d) : 0
    end
    return abs(ee)
end

function make_hermitian(A::ITensor)
    A_inds = ITensors.inds(A)
    @assert length(A_inds) == 2
    return (A + ITensors.swapind(dag(A), first(A_inds), last(A_inds))) / 2
end

function ITensorNetworks.ket_network(bpc::AbstractBeliefPropagationCache)
    return ket_network(tensornetwork(bpc))
end

#Calculate the correlation flowing around single loop of the bp cache via an eigendecomposition
function loop_correlation(bpc::BeliefPropagationCache, loop::Vector{<:PartitionEdge}, target_pe::PartitionEdge)

    is_tree(partitions_graph(bpc)) && return 0
    bpc = copy(bpc)

    pes = vcat(loop, [target_pe])
    incoming_es = boundary_partitionedges(bpc, pes)
    incoming_messages = ITensor[only(message(bpc, pe)) for pe in incoming_es]
    pvs = unique(vcat(src.(loop), dst.(loop)))
    vs = vertices(bpc, pvs)

    src_vs = vertices(bpc, src(target_pe))

    pe_linkinds = linkinds(bpc, target_pe)
    pe_linkinds_sim = sim.(pe_linkinds)

    ts = ITensor[]

    for v in src_vs
        t = bpc[v]
        t_inds = filter(i -> i ∈ pe_linkinds, inds(t))
        if !isempty(t_inds)
            t_ind = only(t_inds)
            t_ind_pos = findfirst(x -> x == t_ind, pe_linkinds)
            t = replaceind(t, t_ind, pe_linkinds_sim[t_ind_pos])
        end
        push!(ts, t)
    end

    ts = ITensor[ts; ITensor[bpc[v] for v in setdiff(vs, src_vs)]]

    tensors = [ts; incoming_messages]
    seq = ITensorNetworks.contraction_sequence(tensors; alg = "einexpr", optimizer = Greedy())
    t = contract(tensors; sequence = seq)

    row_combiner, col_combiner = ITensors.combiner(pe_linkinds), ITensors.combiner(pe_linkinds_sim)
    t = t * row_combiner * col_combiner
    t = array(t)
    λs = reverse(sort(LinearAlgebra.eigvals(t); by = abs))

    err = 1.0 - abs(λs[1]) / sum(abs.(λs))

    return err
end

#Calculate the correlations flowing around each of the primitive loops of the BP cache
function loop_correlations(bpc::BeliefPropagationCache, smallest_loop_size::Int; kwargs...)
    pg = partitions_graph(bpc)
    cycles = NamedGraphs.cycle_to_path.(NamedGraphs.unique_simplecycles_limited_length(pg, smallest_loop_size))
    corrs = []
    for loop in cycles
        corrs = append!(corrs, loop_correlation(bpc, PartitionEdge.(loop[1:(length(loop)-1)]), reverse(PartitionEdge(last(loop))); kwargs...))
    end
    return corrs
end

default_normalize(::Algorithm"squarebp") = true
default_sequence_alg(::Algorithm"squarebp") = "optimal"
function ITensorNetworks.set_default_kwargs(alg::Algorithm"squarebp")
    normalize = get(alg.kwargs, :normalize, default_normalize(alg))
    sequence_alg = get(alg.kwargs, :sequence_alg, default_sequence_alg(alg))
    return Algorithm("squarebp"; normalize, sequence_alg)
end

function initialize_square_bp_messages!(bpc::BeliefPropagationCache)
    dtype = datatype(tensornetwork(bpc))
    for pe in vcat(partitionedges(bpc), reverse.(partitionedges(bpc)))
        lind = only(ITensorNetworks.linkinds(bpc, pe))
        m = adapt(dtype, denseblocks(delta(lind, dag(lind)')))
        ITensorNetworks.set_message!(bpc, pe, ITensor[m])
    end
end

function ITensorNetworks.updated_message(alg::Algorithm"squarebp", bpc::AbstractBeliefPropagationCache, edge::PartitionEdge)
    vertex = src(edge)
    incoming_ms = ITensorNetworks.incoming_messages(bpc, vertex; ignore_edges = PartitionEdge[reverse(edge)])
    state = only(ITensorNetworks.factors(bpc, vertex))
    state_dag = noprime(dag(prime(state)), tags = "Site")
    contract_list = ITensor[incoming_ms; [state, state_dag]]
    sequence = ITensorNetworks.contraction_sequence(contract_list; alg = alg.kwargs.sequence_alg)
    m = make_hermitian(ITensors.contract(contract_list; sequence))
    message_norm = norm(m)
    if alg.kwargs.normalize && !iszero(message_norm)
        m /= message_norm
    end
    return ITensor[m]
end

function ITensorNetworks.siteinds(bpc::BeliefPropagationCache)
    is_flat(bpc) && return siteinds(tensornetwork(bpc))
    return siteinds(ket_network(bpc))
end