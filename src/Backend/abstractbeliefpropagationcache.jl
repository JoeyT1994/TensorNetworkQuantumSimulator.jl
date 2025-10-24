using Graphs: Graphs
using Adapt

abstract type AbstractBeliefPropagationCache{V} <: AbstractGraph{V} end

#Interface
factor(bp_cache::AbstractBeliefPropagationCache, vertex) = not_implemented()
setfactor!(bp_cache::AbstractBeliefPropagationCache, vertex, factor) = not_implemented()
messages(bp_cache::AbstractBeliefPropagationCache) = not_implemented()
message(bp_cache::AbstractBeliefPropagationCache, edge::AbstractEdge) = not_implemented()
function default_message(bp_cache::AbstractBeliefPropagationCache, edge::AbstractEdge)
    return not_implemented()
end
default_messages(bp_cache::AbstractBeliefPropagationCache) = not_implemented()
function setmessage!(bp_cache::AbstractBeliefPropagationCache, edge::AbstractEdge, message)
    return not_implemented()
end
function deletemessage!(bp_cache::AbstractBeliefPropagationCache, edge::AbstractEdge)
    return not_implemented()
end
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

function vertex_scalar(bp_cache::AbstractBeliefPropagationCache, vertex; kwargs...)
    return not_implemented()
end
function edge_scalar(
        bp_cache::AbstractBeliefPropagationCache, edge::AbstractEdge; kwargs...
    )
    return not_implemented()
end

#Graph functionality needed
Graphs.vertices(bp_cache::AbstractBeliefPropagationCache) = not_implemented()
Graphs.edges(bp_cache::AbstractBeliefPropagationCache) = not_implemented()
function NamedGraphs.GraphsExtensions.boundary_edges(
        bp_cache::AbstractBeliefPropagationCache, vertices; kwargs...
    )
    return not_implemented()
end

#Functions derived from the interface
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
        setindex_preserve_graph!(bp_cache, f(network(bp_cache)[v]), v)
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
    rescale!(bpc, args...;kwargs...)
    return bpc
end
