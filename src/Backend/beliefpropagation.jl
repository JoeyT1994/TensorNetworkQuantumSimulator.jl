const _default_bp_update_maxiter = 25
const _default_bp_update_tol = 1e-10

## Frontend functions

# `default_message_update` lives in ITensorNetworks.jl
default_posdef_message_update_function(ms) = make_hermitian.(default_message_update(ms))

function default_posdef_bp_update_kwargs(; cache_is_tree = false)
    message_update_function = default_posdef_message_update_function
    return (; maxiter=default_bp_update_maxiter(cache_is_tree), tol=_default_bp_update_tol, message_update_kwargs=(; message_update_function))
end

function default_nonposdef_bp_update_kwargs(; cache_is_tree = false)
    message_update_function = default_message_update
    return (;  maxiter=default_bp_update_maxiter(cache_is_tree), tol=_default_bp_update_tol, message_update_kwargs=(; message_update_function))
end

function default_bp_update_maxiter(cache_is_tree::Bool = false)
    !cache_is_tree && return _default_bp_update_maxiter
    return 1
end

"""
    updatecache(bp_cache::BeliefPropagationCache; maxiter::Int64, tol::Number, message_update_kwargs = (; message_update_function = default_message_update))

Update the message tensors inside a bp-cache, running over the graph up to maxiter times until convergence to the desired tolerance `tol`.
If the cache is positive definite, the message update function can
"""
function updatecache(bp_cache; maxiter=default_bp_update_maxiter(is_tree(partitioned_graph(bp_cache))), tol=_default_bp_update_tol, message_update_kwargs=(; message_update_function=default_message_update))
    return update(bp_cache; maxiter, tol, message_update_kwargs)
end

"""
    build_bp_cache(ψ::ITensorNetwork, args...; kwargs...)

Build the tensornetwork and cache of message tensors for the norm square network `ψIψ`.
"""
function build_bp_cache(
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
        if parent(p1) ∉ neighbors(partitioned_graph(bpc), parent(p2))
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
    cache = isnothing(cache!) ? build_bp_cache(ψ; cache_update_kwargs) : cache![]
    ψ_vidal = VidalITensorNetwork(ψ; cache)
    bt = ITensorNetworks.bond_tensor(ψ_vidal, e)
    ee = 0
    for d in diag(bt)
        ee -= abs(d) >= eps(eltype(bt)) ? d * d * log2(d * d) : 0
    end
    return abs(ee)
end

function make_hermitian(A::ITensor)
    A_inds = inds(A)
    @assert length(A_inds) == 2
    return (A + ITensors.swapind(conj(A), first(A_inds), last(A_inds))) / 2
end

#Delete the message tensor on partition edge pe from the cache
function delete_message!(bpc::AbstractBeliefPropagationCache, pe::PartitionEdge)
    return delete_messages!(bpc, [pe])
end

#Delete the message tensors on the vector of partition edges pes from the cache
function delete_messages!(bpc::AbstractBeliefPropagationCache, pes::Vector)
    ms = messages(bpc)
    for pe in pes
        haskey(ms, pe) && delete!(ms, pe)
    end
    return bpc
end