using Graphs: Graphs
using Adapt

abstract type AbstractBeliefPropagationCache{V} <: AbstractNamedGraph{V} end

#Interface
messages(bp_cache::AbstractBeliefPropagationCache) = not_implemented()
contraction_sequences(bp_cache::AbstractBeliefPropagationCache) = not_implemented()
default_messages() = Dictionary{NamedEdge, Union{Tensor, Vector{Tensor}}}()

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
    fast = norm_scalar_kernel(network(bp_cache), [vertex], incoming_ms; op_strings = v -> "I")
    fast !== nothing && return fast
    return scalar(contract_optimal([bp_factors(bp_cache, vertex); incoming_ms]))
end

function edge_scalar(
        bp_cache::AbstractBeliefPropagationCache, edge::AbstractEdge; kwargs...
    )
    return not_implemented()
end

network(bp_cache::AbstractBeliefPropagationCache) = not_implemented()
graph(bp_cache::AbstractBeliefPropagationCache) = not_implemented()

#Forward onto the network
for f in [
        :(bp_factors),
        :(default_bp_maxiter),
        :(virtualinds),
        :datatype,
        :scalartype,
        :(maxvirtualdim),
        :(default_message),
        :(siteinds),
    ]
    @eval begin
        function $f(bp_cache::AbstractBeliefPropagationCache, args...; kwargs...)
            return $f(network(bp_cache), args...; kwargs...)
        end
    end
end

function invalidate_contraction_sequences!(bp_cache::AbstractBeliefPropagationCache)
    seq_cache = contraction_sequences(bp_cache)
    !isnothing(seq_cache) && empty!(seq_cache)
    return bp_cache
end

function setindex_preserve!(bp_cache::AbstractBeliefPropagationCache, value::Tensor, vertex)
    setindex_preserve!(network(bp_cache), value, vertex)
    return bp_cache
end

#Forward onto the graph
for f in [
        :(NamedGraphs.edgetype),
        :(NamedGraphs.vertices),
        :(NamedGraphs.edges),
        :(NamedGraphs.steiner_tree),
        :(NamedGraphs.is_tree),
    ]
    @eval begin
        function $f(bp_cache::AbstractBeliefPropagationCache, args...; kwargs...)
            return $f(graph(bp_cache), args...; kwargs...)
        end
    end
end

NamedGraphs.encoded_vertex(bp_cache::AbstractBeliefPropagationCache, vertex) = NamedGraphs.encoded_vertex(graph(bp_cache), vertex)
NamedGraphs.decoded_vertex(bp_cache::AbstractBeliefPropagationCache, code::Integer) = NamedGraphs.decoded_vertex(graph(bp_cache), code)
NamedGraphs.encoded_graph(bp_cache::AbstractBeliefPropagationCache) = NamedGraphs.encoded_graph(graph(bp_cache))

#Functions derived from the interface
function deletemessage!(bp_cache::AbstractBeliefPropagationCache, e::AbstractEdge)
    ms = messages(bp_cache)
    delete!(ms, e)
    return bp_cache
end

function setmessage!(bp_cache::AbstractBeliefPropagationCache, e::AbstractEdge, message::Union{Tensor, Vector{<:Tensor}})
    ms = messages(bp_cache)
    set!(ms, e, message)
    return bp_cache
end

function message(bp_cache::AbstractBeliefPropagationCache, edge::AbstractEdge; kwargs...)
    ms = messages(bp_cache)
    return get(() -> default_message(bp_cache, edge; kwargs...), ms, edge)
end

function messages(bp_cache::AbstractBeliefPropagationCache, edges::Vector{<:AbstractEdge})
    ms = Tensor[]
    for e in edges
        m = message(bp_cache, e)
        if m isa Tensor
            push!(ms, m)
        else
            append!(ms, m)
        end
    end
    return ms
end

function setmessages!(bp_cache::AbstractBeliefPropagationCache, edges, messages)
    for (e, m) in zip(edges, messages)
        setmessage!(bp_cache, e, m)
    end
    return bp_cache
end

function deletemessages!(
        bp_cache::AbstractBeliefPropagationCache, edges::Vector{<:AbstractEdge} = collect(edges(bp_cache))
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
    b_edges = NamedGraphs.boundary_edges(bp_cache, vertices; dir = :in)
    b_edges = !isempty(ignore_edges) ? setdiff(b_edges, ignore_edges) : b_edges
    return messages(bp_cache, b_edges)
end

function incoming_messages(bp_cache::AbstractBeliefPropagationCache, vertex; kwargs...)
    return incoming_messages(bp_cache, [vertex]; kwargs...)
end

# The vertices an observable on `vs` is closed on: `vs` joined by a Steiner tree.
function observable_region(bp_cache::AbstractBeliefPropagationCache, vs)
    return length(vs) == 1 ? vs : collect(vertices(steiner_tree(network(bp_cache), vs)))
end

# The norm network on `region`, with operators from `op_strings`, contracted against its environment.
function contract_region(
        bp_cache::AbstractBeliefPropagationCache, region, op_strings::Function;
        incoming_ms = incoming_messages(bp_cache, region)
    )
    #TODO: If there are a lot of tensors here, (more than 100 say), we need to think about defining a custom sequence as optimal may be too slow
    return contract_optimal([norm_factors(network(bp_cache), region; op_strings); incoming_ms])
end

# `contract_region` as a scalar, through the backend's fused kernel where it has one.
function region_scalar(bp_cache::AbstractBeliefPropagationCache, region, op_strings::Function)
    incoming_ms = incoming_messages(bp_cache, region)
    fast = norm_scalar_kernel(network(bp_cache), region, incoming_ms; op_strings)
    fast !== nothing && return fast
    return scalar(contract_region(bp_cache, region, op_strings; incoming_ms))
end

# The message on `edge` recomputed from the cache. The same list recurs every sweep, so its
# contraction order is memoised in the cache; `update` and `update_partition!` clear the memo when
# they are done.
function updated_message(
        alg::Algorithm"contract", bp_cache::AbstractBeliefPropagationCache, edge::NamedEdge
    )
    vertex = src(edge)
    incoming_ms = incoming_messages(bp_cache, vertex; ignore_edges = (reverse(edge),))
    m = norm_message_kernel(network(bp_cache), vertex, incoming_ms)
    if isnothing(m)
        ts = Tensor[incoming_ms; bp_factors(bp_cache, vertex)]
        sequence = get!(() -> contraction_sequence(ts; alg = alg.kwargs.sequence_alg), contraction_sequences(bp_cache), vertex => edge)
        m = contract(ts; sequence)
    end
    return alg.kwargs.normalize ? normalize_message(m) : m
end

function update_message!(alg::Algorithm"contract", bp_cache::AbstractBeliefPropagationCache, edge::AbstractEdge)
    return setmessage!(bp_cache, edge, updated_message(alg, bp_cache, edge))
end

# Unit 2-norm (the element sum can cancel for signed or complex networks), with the phase of the
# sum divided out so that roundoff phases do not accumulate around loops.
function normalize_message(m)
    n = norm(m)
    iszero(n) && return m
    s = sum(m)
    phase = abs(s) > sqrt(eps(real(typeof(s)))) * n ? s / abs(s) : one(s)
    return m / (n * phase)
end

function updated_message(
        bp_cache::AbstractBeliefPropagationCache,
        edge::NamedEdge;
        alg = default_message_update_alg(bp_cache),
        kwargs...,
    )
    return updated_message(set_default_kwargs(Algorithm(alg; kwargs...), bp_cache), bp_cache, edge)
end

"""
    update_iteration!(alg::Algorithm"bp", bpc, edges; diff = false)

Update the messages on `edges` in sequence. Returns the summed change of the messages if `diff`,
otherwise `nothing`.
"""
function update_iteration!(alg::Algorithm"bp", bpc::AbstractBeliefPropagationCache, edges::Vector; diff::Bool = false)
    total = 0.0
    for e in edges
        previous = diff ? message(bpc, e) : nothing
        update_message!(alg.kwargs.message_update_alg, bpc, e)
        diff && (total += message_diff(message(bpc, e), previous))
    end
    return diff ? total : nothing
end

# Repeat `step!()`, which returns how much it changed things (or `nothing`), up to `maxiter` times,
# stopping once the change is at most `tolerance` (never, if `tolerance` is `nothing`). Reports the
# outcome as `name` and returns the number of steps taken.
function iterate_to_convergence(step!, maxiter::Integer, tolerance; name, verbose::Bool = false)
    change = nothing
    for i in 1:maxiter
        change = step!()
        if !isnothing(tolerance) && !isnothing(change) && change <= tolerance
            verbose && println("$name converged after $i iterations.")
            return i
        end
    end
    if !isnothing(tolerance)
        msg = "$name did not converge to tolerance $tolerance after $maxiter iterations (final change: $change)."
        verbose ? println(msg) : @warn(msg)
    end
    return maxiter
end

function update(alg::Algorithm"bp", bpc::AbstractBeliefPropagationCache)
    isnothing(alg.kwargs.maxiter) && error("You need to specify a number of iterations for BP!")
    bpc = copy(bpc)
    invalidate_contraction_sequences!(bpc)
    es = alg.kwargs.edge_sequence
    diff = !isnothing(alg.kwargs.tolerance)
    # the change per sweep is the average message change
    step!() = (d = update_iteration!(alg, bpc, es; diff); diff ? d / length(es) : nothing)
    # the fused kernels share one set of scratch buffers across all the sweeps
    with_kernel_workspace() do
        iterate_to_convergence(step!, alg.kwargs.maxiter, alg.kwargs.tolerance; name = "BP", verbose = alg.kwargs.verbose)
    end
    invalidate_contraction_sequences!(bpc)
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

# The logarithm, on the principal branch for negative reals.
_log(z) = real(z) < 0 ? log(complex(z)) : log(z)

"""
    freenergy(cache)

The free energy `-ln Z` of the network the cache was built from, `Z` being its contraction as the
cache estimates it (for BP and boundary MPS, the ratio of its vertex to its edge scalars).
"""
function freenergy(bp_cache::AbstractBeliefPropagationCache)
    numerator_terms, denominator_terms = scalar_factors_quotient(bp_cache)
    any(iszero, denominator_terms) && return -Inf
    return sum(_log, denominator_terms) - sum(_log, numerator_terms)
end

"""
    partitionfunction(cache)

The contraction `Z = exp(-freenergy(cache))` of the network the cache was built from.
"""
partitionfunction(cache::AbstractBeliefPropagationCache) = exp(-freenergy(cache))

function rescale_messages!(bp_cache::AbstractBeliefPropagationCache, edge::AbstractEdge)
    return rescale_messages!(bp_cache, [edge])
end

function rescale_messages!(bp_cache::AbstractBeliefPropagationCache)
    return rescale_messages!(bp_cache, collect(edges(bp_cache)))
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
