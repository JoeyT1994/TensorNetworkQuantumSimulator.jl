using Graphs: Graphs
using Adapt

abstract type AbstractBeliefPropagationCache{V} <: AbstractNamedGraph{V} end

#Interface
messages(bp_cache::AbstractBeliefPropagationCache) = not_implemented()
default_messages() = Dictionary{NamedEdge, Union{ITensor, Vector{ITensor}}}()

function rescale_messages!(
        bp_cache::AbstractBeliefPropagationCache, edges::Vector{<:AbstractEdge}; kwargs...
    )
    return not_implemented()
end
function rescale_vertices!(
        bp_cache::AbstractBeliefPropagationCache, vertices::Vector; kwargs...
    )
    return not_implemented()
end

function vertex_scalar(bp_cache::AbstractBeliefPropagationCache, vertex)
    incoming_ms = incoming_messages(bp_cache, vertex)
    state = bp_factors(bp_cache, vertex)
    contract_list = [state; incoming_ms]
    sequence = contraction_sequence(contract_list; alg = "optimal")
    return contract(contract_list; sequence)[]
end

function edge_scalar(
        bp_cache::AbstractBeliefPropagationCache, edge::AbstractEdge; kwargs...
    )
    return not_implemented()
end

network(bp_cache::AbstractBeliefPropagationCache) = not_implemented()

#Forward onto the network
for f in [
        :(graph),
        :(bp_factors),
        :(default_bp_maxiter),
        :(virtualinds),
        :(ITensors.datatype),
        :(ITensors.NDTensors.scalartype),
        :(maxvirtualdim),
        :(default_message),
        :(siteinds),
        :(setindex_preserve!),
        :(NamedGraphs.edgetype),
        :(NamedGraphs.vertices),
        :(NamedGraphs.edges),
        :(NamedGraphs.position_graph),
        :(NamedGraphs.ordered_vertices),
        :(NamedGraphs.vertex_positions),
        :(NamedGraphs.steiner_tree),
        :(NamedGraphs.is_tree),
    ]
    @eval begin
        function $f(bp_cache::AbstractBeliefPropagationCache, args...; kwargs...)
            return $f(network(bp_cache), args...; kwargs...)
        end
    end
end

#Functions derived from the interface
function deletemessage!(bp_cache::AbstractBeliefPropagationCache, e::AbstractEdge)
    ms = messages(bp_cache)
    delete!(ms, e)
    return bp_cache
end

function setmessage!(bp_cache::AbstractBeliefPropagationCache, e::AbstractEdge, message::Union{ITensor, Vector{<:ITensor}})
    ms = messages(bp_cache)
    set!(ms, e, message)
    return bp_cache
end

function message(bp_cache::AbstractBeliefPropagationCache, edge::AbstractEdge; kwargs...)
    ms = messages(bp_cache)
    return get(() -> default_message(bp_cache, edge; kwargs...), ms, edge)
end

function messages(bp_cache::AbstractBeliefPropagationCache, edges::Vector{<:AbstractEdge})
    isempty(edges) && return ITensor[]
    ms = reduce(vcat, [message(bp_cache, e) for e in edges])
    ms isa ITensor && return ITensor[ms]
    return ms
end

function setmessages!(bp_cache::AbstractBeliefPropagationCache, edges, messages)
    for (e, m) in zip(edges)
        setmessage!(bp_cache, e, m)
    end
    return
end

function deletemessages!(
        bp_cache::AbstractBeliefPropagationCache, edges::Vector{<:AbstractEdge} = edges(bp_cache)
    )
    for e in edges
        deletemessage!(bp_cache, e)
    end
    return bp_cache
end

function vertex_scalars(
        bp_cache::AbstractBeliefPropagationCache, vertices = collect(Graphs.vertices(bp_cache)); kwargs...
    )
    return map(v -> vertex_scalar(bp_cache, v; kwargs...), vertices)
end

function edge_scalars(
        bp_cache::AbstractBeliefPropagationCache, edges = Graphs.edges(bp_cache); kwargs...
    )
    return map(e -> edge_scalar(bp_cache, e; kwargs...), edges)
end

function scalar_factors_quotient(bp_cache::AbstractBeliefPropagationCache)
    return vertex_scalars(bp_cache), edge_scalars(bp_cache)
end

function incoming_messages(
        bp_cache::AbstractBeliefPropagationCache, vertices::Vector{<:Any}; ignore_edges = []
    )
    b_edges = NamedGraphs.GraphsExtensions.boundary_edges(bp_cache, vertices; dir = :in)
    b_edges = !isempty(ignore_edges) ? setdiff(b_edges, ignore_edges) : b_edges
    return messages(bp_cache, b_edges)
end

function incoming_messages(bp_cache::AbstractBeliefPropagationCache, vertex; kwargs...)
    return incoming_messages(bp_cache, [vertex]; kwargs...)
end

function updated_message(
        alg::Algorithm"contract", bp_cache::AbstractBeliefPropagationCache, edge::NamedEdge
    )
    vertex = src(edge)
    incoming_ms = incoming_messages(
        bp_cache, vertex; ignore_edges = typeof(edge)[reverse(edge)]
    )
    state = bp_factors(bp_cache, vertex)
    contract_list = ITensor[incoming_ms; state]
    sequence = contraction_sequence(contract_list; alg = alg.kwargs.sequence_alg)
    updated_message = contract(contract_list; sequence)

    if alg.kwargs.enforce_hermiticity
        updated_message = make_hermitian(updated_message)
    end

    if alg.kwargs.normalize
        message_norm = LinearAlgebra.norm(updated_message)
        if !iszero(message_norm)
            updated_message /= message_norm
        end
    end

    return updated_message
end

function updated_message(
        bp_cache::AbstractBeliefPropagationCache,
        edge::NamedEdge;
        alg = default_message_update_alg(bp_cache),
        kwargs...,
    )
    return updated_message(set_default_kwargs(Algorithm(alg; kwargs...)), bp_cache, edge)
end

"""
Do a sequential update of the message tensors on `edges`
"""
function update_iteration!(
        alg::Algorithm"bp",
        bpc::AbstractBeliefPropagationCache,
        edges::Vector;
        (update_diff!) = nothing,
    )
    for e in edges
        prev_message = !isnothing(update_diff!) ? message(bpc, e) : nothing
        update_message!(alg.kwargs.message_update_alg, bpc, e)
        if !isnothing(update_diff!)
            update_diff![] += message_diff(message(bpc, e), prev_message)
        end
    end
    return bpc
end

"""
More generic interface for update, with default params
"""
function update(alg::Algorithm"bp", bpc::AbstractBeliefPropagationCache)
    compute_error = !isnothing(alg.kwargs.tolerance)
    if isnothing(alg.kwargs.maxiter)
        error("You need to specify a number of iterations for BP!")
    end
    bpc = copy(bpc)
    for i in 1:alg.kwargs.maxiter
        diff = compute_error ? Ref(0.0) : nothing
        update_iteration!(alg, bpc, alg.kwargs.edge_sequence; (update_diff!) = diff)
        if compute_error && (diff.x / length(alg.kwargs.edge_sequence)) <= alg.kwargs.tolerance
            if alg.kwargs.verbose
                println("BP converged to desired precision after $i iterations.")
            end
            break
        end
    end
    return bpc
end

function update(bpc::AbstractBeliefPropagationCache; alg = default_update_alg(bpc), kwargs...)
    return update(set_default_kwargs(Algorithm(alg; kwargs...), bpc), bpc)
end

#Adapt interface for changing device
function map_messages(f, bp_cache::AbstractBeliefPropagationCache, es = keys(messages(bp_cache)))
    bp_cache = copy(bp_cache)
    for e in es
        setmessage!(bp_cache, e, f(message(bp_cache, e)))
    end
    return bp_cache
end
function map_factors(f, bp_cache::AbstractBeliefPropagationCache, vs = vertices(bp_cache))
    bp_cache = copy(bp_cache)
    for v in vs
        setindex_preserve!(bp_cache, f(network(bp_cache)[v]), v)
    end
    return bp_cache
end
function adapt_messages(to, bp_cache::AbstractBeliefPropagationCache, args...)
    return map_messages(adapt(to), bp_cache, args...)
end
function adapt_factors(to, bp_cache::AbstractBeliefPropagationCache, args...)
    return map_factors(adapt(to), bp_cache, args...)
end

function Adapt.adapt_structure(to, bpc::AbstractBeliefPropagationCache)
    bpc = adapt_messages(to, bpc)
    bpc = adapt_factors(to, bpc)
    return bpc
end

function freenergy(bp_cache::AbstractBeliefPropagationCache)
    numerator_terms, denominator_terms = scalar_factors_quotient(bp_cache)
    if any(t -> real(t) < 0, numerator_terms)
        numerator_terms = complex.(numerator_terms)
    end
    if any(t -> real(t) < 0, denominator_terms)
        denominator_terms = complex.(denominator_terms)
    end

    any(iszero, denominator_terms) && return -Inf
    return sum(log.(numerator_terms)) - sum(log.((denominator_terms)))
end

function partitionfunction(bp_cache::AbstractBeliefPropagationCache)
    return exp(freenergy(bp_cache))
end

function rescale_messages!(bp_cache::AbstractBeliefPropagationCache, edge::AbstractEdge)
    return rescale_messages!(bp_cache, [edge])
end

function rescale_messages!(bp_cache::AbstractBeliefPropagationCache)
    return rescale_messages!(bp_cache, edges(bp_cache))
end

function rescale_vertices!(bpc::AbstractBeliefPropagationCache; kwargs...)
    return rescale_vertices!(bpc, collect(vertices(bpc)); kwargs...)
end

function rescale!(bpc::AbstractBeliefPropagationCache, args...; kwargs...)
    rescale_messages!(bpc)
    rescale_vertices!(bpc, args...; kwargs...)
    return bpc
end

function rescale(bpc::AbstractBeliefPropagationCache, args...; kwargs...)
    bpc = copy(bpc)
    rescale!(bpc, args...; kwargs...)
    return bpc
end
