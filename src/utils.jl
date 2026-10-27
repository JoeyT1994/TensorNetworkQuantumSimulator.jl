# Boundary MPS helpers for checking graph formats
function is_line_graph(g::AbstractGraph)
    vs = collect(vertices(g))
    nvs = length(vs)
    length(vs) == 1 && return true
    !is_tree(g) && return false
    ds = sort([degree(g, v) for v in vs])
    ds != vcat([1, 1], [2 for d in 1:(nvs - 2)]) && return false
    return true
end

function is_ring_graph(g::AbstractGraph)
    isempty(edges(g)) && return false
    g_mod = rem_edge(g, first(edges(g)))
    return is_line_graph(g_mod)
end

# The square root and pseudo-inverse square root of a Hermitian two-leg tensor, with eigenvalues
# at or below `cutoff` in magnitude dropped.
function pseudo_sqrt_inv_sqrt(M::Tensor; cutoff = 10 * eps(real(scalartype(M))))
    @assert length(inds(M)) == 2
    l, r = inds(M)
    D, U = safe_eigen(M, l, r; ishermitian = true)
    Ul = replaceinds(U, [r, commonind(D, U)], [l, noncommonind(D, U)])
    root(f) = Ul * map_diag(x -> abs(x) <= cutoff ? zero(x) : f(x), D) * dag(U)
    return root(sqrt), root(inv ∘ sqrt)
end

# Adapt `t` to the storage datatype (eltype + device) of `ref`.
adapt_like(ref, t) = adapt(datatype(ref))(t)

function identity_tensor(eltype, row_inds::Vector{<:Index}, col_inds::Vector{<:Index})
    c_row, c_col = combiner(row_inds), combiner(col_inds)
    t = dense(delta(eltype, combinedind(c_row), combinedind(c_col)))
    return (t * c_row) * c_col
end

# The cache each algorithm works on (or is built from a TensorNetworkState), and the
# functionality it does not support.
const ALGORITHM_CACHES = Dict(
    "exact" => Any,
    "bp" => BeliefPropagationCache,
    "loopcorrections" => BeliefPropagationCache,
    "boundarymps" => BoundaryMPSCache,
    "ctmrg" => CTMRGCache,
)
const ALGORITHM_UNSUPPORTED = Dict(
    "exact" => ("normalize", "sample", "sample_certified", "truncate", "edge_entropy"),
    "bp" => ("sample_certified",),
    "loopcorrections" => ("normalize", "expect", "sample", "sample_certified", "truncate", "rdm", "edge_entropy"),
    "boundarymps" => ("normalize", "edge_entropy"),
    "ctmrg" => ("normalize", "sample", "sample_certified", "truncate", "edge_entropy"),
)

# Check that `alg` can do `f` on `tns`.
function algorithm_check(tns::Union{AbstractBeliefPropagationCache, TensorNetworkState}, f::String, alg)
    haskey(ALGORITHM_CACHES, alg) ||
        error("Unrecognized algorithm $(repr(alg)). Must be one of $(join(repr.(sort(collect(keys(ALGORITHM_CACHES)))), ", ")).")
    cache = ALGORITHM_CACHES[alg]
    tns isa Union{cache, TensorNetworkState} ||
        error("Expected $cache or TensorNetworkState for the $(repr(alg)) algorithm, got $(typeof(tns)).")
    alg == "bp" && f == "sample_certified" &&
        error("Certified sampling needs an estimate of p(x)/q(x), which the 'bp' sampler does not produce. Use alg = 'boundarymps'.")
    f in ALGORITHM_UNSUPPORTED[alg] && error("The $(repr(alg)) algorithm does not support $f yet.")
    return nothing
end

default_alg(bp_cache::BeliefPropagationCache) = "bp"
default_alg(bmps_cache::BoundaryMPSCache) = "boundarymps"
default_alg(ctm_cache::CTMRGCache) = "ctmrg"
default_alg(any) = error("You must specify a contraction algorithm. Currently supported: exact, bp, boundarymps and ctmrg.")

# Build the cache `alg` needs over `network` and run it to convergence. Shared by the state-level
# entry points of `expect`, `reduced_density_matrix`, `norm_sqr`, `inner` and `normalize`, which all
# wrap a network in a cache, converge it, then hand off to the corresponding cache-level method.
# `cache_update_kwargs` is deliberately required: each entry point documents its own default.
default_cache_update_kwargs(::Union{Algorithm"bp", Algorithm"loopcorrections"}, network) = default_bp_update_kwargs(network)
default_cache_update_kwargs(::Algorithm"boundarymps", network) = default_bmps_update_kwargs(network)
default_cache_update_kwargs(::Algorithm"ctmrg", network) = (;)

# A contraction (a norm, an overlap, a partition function) only needs CTMRG's free energy
# converged, not its messages.
scalar_cache_update_kwargs(::Algorithm, kwargs) = kwargs
scalar_cache_update_kwargs(::Algorithm"ctmrg", kwargs) = merge((; convergence = :freenergy), kwargs)

function converged_cache(
        ::Union{Algorithm"bp", Algorithm"loopcorrections"}, network;
        cache_update_kwargs,
    )
    return update(BeliefPropagationCache(network); cache_update_kwargs...)
end

function converged_cache(
        ::Algorithm"ctmrg", network;
        maxdim::Integer,
        cache_update_kwargs,
        cache_kwargs...,
    )
    return update(CTMRGCache(network, maxdim; cache_kwargs...); cache_update_kwargs...)
end

function converged_cache(
        ::Algorithm"boundarymps", network;
        mps_bond_dimension::Integer,
        partition_by = "row",
        gauge_state = false,
        cache_update_kwargs,
    )
    cache = BoundaryMPSCache(network, mps_bond_dimension; partition_by, gauge_state)
    return update(cache; cache_update_kwargs...)
end

"""
    safe_eigen(m::Tensor, args...; kwargs...)

The eigendecomposition of `m` (the backend's `eigen`) in at least double precision, for numerical
stability. Lower-precision data is promoted on the host and the factors are returned in `m`'s storage.
"""
function safe_eigen(m::Tensor, args...; kwargs...)
    T = promote_type(eltype(m), Float64)
    T == eltype(m) && return eigen(m, args...; kwargs...)
    D, U = eigen(adapt(Vector{T}, m), args...; kwargs...)
    return adapt_like(m, D), adapt_like(m, U)
end

collect_vertices(e::NamedEdge, g::NamedGraph) = collect_vertices([src(e), dst(e)], g)

collect_vertices(es::Vector{<:NamedEdge}, g::NamedGraph) = reduce(vcat, [collect_vertices(e, g) for e in es])

# Levenshtein edit distance between two strings.
function levenshtein(a::AbstractString, b::AbstractString)
    av, bv = collect(a), collect(b)
    m, n = length(av), length(bv)
    m == 0 && return n
    n == 0 && return m
    prev = collect(0:n)
    curr = zeros(Int, n + 1)
    for i in 1:m
        curr[1] = i
        for j in 1:n
            cost = av[i] == bv[j] ? 0 : 1
            curr[j + 1] = min(
                curr[j] + 1,        # insertion
                prev[j + 1] + 1,    # deletion
                prev[j] + cost,     # substitution
            )
        end
        prev, curr = curr, prev
    end
    return prev[n + 1]
end

function collect_vertices(verts, g::NamedGraph)
    vt = vertextype(g)

    if vt == Any
        if verts isa AbstractVector
            return verts
        else
            return [verts]
        end
    end

    verts isa vt && return [verts]
    collected_verts = vt[]
    for v in verts
        if v isa vt
            push!(collected_verts, v)
        else
            error("Vertex does not match the vertex type of the tensor network")
        end
    end

    length(unique(collected_verts)) != length(collected_verts) && error("Repeated vertex in collection")
    return collected_verts
end
