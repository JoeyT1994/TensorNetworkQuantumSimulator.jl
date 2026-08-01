using Graphs: Graphs
using Adapt
using ITensors: Algorithm

abstract type AbstractBeliefPropagationCache{V} <: AbstractNamedGraph{V} end

#Interface
messages(bp_cache::AbstractBeliefPropagationCache) = not_implemented()
contraction_sequences(bp_cache::AbstractBeliefPropagationCache) = not_implemented()
default_messages() = Dictionary{NamedEdge, Union{ITensor, Vector{ITensor}}}()

function rescale_messages!(
        bp_cache::AbstractBeliefPropagationCache, edges::Vector{<:AbstractEdge}; kwargs...
    )
    return not_implemented()
end
# Everything here is on the interface -- `vertex_scalar` goes through `incoming_messages`, which
# a cache may override -- so this is correct for any cache, not just the serial one.
function rescale_vertices!(bp_cache::AbstractBeliefPropagationCache, vertices::Vector)
    tn = network(bp_cache)
    for v in vertices
        vn = vertex_scalar(bp_cache, v)
        s = isreal(vn) ? sign(vn) : one(vn)
        # The scalars are folded before they touch the tensor. `tn[v] * s * inv(...)`
        # left-associates into two separate whole-tensor products, each one allocating a full
        # copy of the site tensor -- and the network's own reference is still live, so that is
        # three factor-sized arrays where one suffices.
        if tn isa TensorNetworkState
            setindex_preserve!(tn, tn[v] * (s * inv(sqrt(vn))), v)
        elseif tn isa TensorNetwork
            setindex_preserve!(tn, tn[v] * (s * inv(vn)), v)
        else
            error("Don't know how to rescale the vertices of this type")
        end
    end
    return bp_cache
end

function vertex_scalar(bp_cache::AbstractBeliefPropagationCache, vertex)
    # Mathematically identical to the generic route below and bounded in peak memory, so it is
    # taken whenever it applies rather than being opt-in: unlike the message update there is no
    # algorithm keyword reaching here, and this is the call that dominates `freenergy` /
    # `norm_sqr` / `inner` on a factor-sized vertex. Returns `nothing` on anything it does not
    # specialise. See `blocked_vertex_scalar`.
    fast = blocked_vertex_scalar(bp_cache, vertex)
    isnothing(fast) || return fast

    incoming_ms = incoming_messages(bp_cache, vertex)
    state = bp_factors(bp_cache, vertex)
    contract_list = [state; incoming_ms]
    sequence = contraction_sequence(contract_list; alg = "optimal")
    return scalar(contract(contract_list; sequence))
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
        :(ITensors.datatype),
        :(ITensors.NDTensors.scalartype),
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

function setindex_preserve!(bp_cache::AbstractBeliefPropagationCache, value::ITensor, vertex)
    setindex_preserve!(network(bp_cache), value, vertex)
    return bp_cache
end

#Forward onto the graph
for f in [
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
            return $f(graph(bp_cache), args...; kwargs...)
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
    ms = ITensor[]
    for e in edges
        m = message(bp_cache, e)
        if m isa ITensor
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

# The graph the messages live on. Normally the network's own graph, but a cache whose messages
# arrive from elsewhere (see `BeliefPropagationCacheMPI`) overrides this with a graph carrying
# extra ghost vertices, and every message lookup below then follows.
messages_graph(bp_cache::AbstractBeliefPropagationCache) = graph(bp_cache)

function incoming_messages(
        bp_cache::AbstractBeliefPropagationCache, vertices::Vector{<:Any}; ignore_edges = []
    )
    b_edges = NamedGraphs.GraphsExtensions.boundary_edges(
        messages_graph(bp_cache), vertices; dir = :in
    )
    b_edges = !isempty(ignore_edges) ? setdiff(b_edges, ignore_edges) : b_edges
    return messages(bp_cache, b_edges)
end

# Dispatches on the cache rather than forwarding, so that a cache which overrides
# `incoming_messages` (or `messages_graph`) is the one `updated_message` sees.
function update_message!(
        message_update_alg::Algorithm, bp_cache::AbstractBeliefPropagationCache, edge::AbstractEdge
    )
    m, (cache_key, sequence, seq_changed) = updated_message(message_update_alg, bp_cache, edge)
    seq_changed && set!(contraction_sequences(bp_cache), cache_key, sequence)
    return setmessage!(bp_cache, edge, m)
end

function incoming_messages(bp_cache::AbstractBeliefPropagationCache, vertex; kwargs...)
    return incoming_messages(bp_cache, [vertex]; kwargs...)
end

function updated_message(
        alg::Algorithm"contract", bp_cache::AbstractBeliefPropagationCache, edge::NamedEdge
    )
    vertex = src(edge)
    incoming_ms = incoming_messages(
        bp_cache, vertex; ignore_edges = (reverse(edge),)
    )
    state = bp_factors(bp_cache, vertex)
    contract_list = ITensor[incoming_ms; state]
    cache_key = vertex => edge
    seq_cache = contraction_sequences(bp_cache)
    seq_changed = false
    if haskey(seq_cache, cache_key)
        sequence = seq_cache[cache_key]
    else
        sequence = contraction_sequence(contract_list; alg = alg.kwargs.sequence_alg)
        seq_changed = true
    end
    updated_message = contract(contract_list; sequence)

    if alg.kwargs.normalize
        message_norm = sum(updated_message)
        if !iszero(message_norm)
            updated_message = updated_message / message_norm
        end
    end

    return updated_message, (cache_key, sequence, seq_changed)
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
# Hooks a distributed cache overrides so it does not have to fork the sweep loop below.
# `sync_messages!` runs at the end of every sweep; `diff_denominator` turns the accumulated diff
# into an average (the count has to be global, or ranks disagree about convergence and the first
# one to stop leaves the rest blocked); `reports_convergence` keeps the log to one rank.
sync_messages!(bpc::AbstractBeliefPropagationCache) = bpc
diff_denominator(bpc::AbstractBeliefPropagationCache, edges) = length(edges)
reports_convergence(bpc::AbstractBeliefPropagationCache) = true
# Sums this rank's contribution into the global one. Identity for a cache that holds everything.
reduce_diff(bpc::AbstractBeliefPropagationCache, diff) = diff

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
    return sync_messages!(bpc)
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
    invalidate_contraction_sequences!(bpc)
    converged = false
    avg_diff = nothing
    niter = alg.kwargs.maxiter
    ndiff = diff_denominator(bpc, alg.kwargs.edge_sequence)
    for i in 1:alg.kwargs.maxiter
        diff = compute_error ? Ref(0.0) : nothing
        update_iteration!(alg, bpc, alg.kwargs.edge_sequence; (update_diff!) = diff)
        if compute_error
            avg_diff = iszero(ndiff) ? 0.0 : reduce_diff(bpc, diff.x) / ndiff
            if avg_diff <= alg.kwargs.tolerance
                converged = true
                niter = i
                break
            end
        end
    end
    if compute_error && reports_convergence(bpc)
        if converged
            alg.kwargs.verbose && println("BP converged to desired precision after $niter iterations.")
        else
            msg = "BP did not converge to tolerance $(alg.kwargs.tolerance) after $niter iterations (final average message change: $avg_diff)."
            alg.kwargs.verbose ? println(msg) : @warn(msg)
        end
    end
    invalidate_contraction_sequences!(bpc)
    release_message_scratch!(bpc)
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
