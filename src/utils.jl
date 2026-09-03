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

#Fermionic messages carry a per-message parity gauge (odd-sector sign); square roots
#need the PSD representative. Identity for non-fermionic backends; see psd_gauge in
#ftensor.jl.
parity_message_gauge(M) = M
parity_message_gauge(M::Tensors.GradedTensor) = Tensors.psd_gauge(M)

function _psd_root_eigenvalue(x, cutoff, inverse::Bool)
    λ = real(x)
    λ > cutoff || return zero(x)
    root = sqrt(λ)
    return oftype(x, inverse ? inv(root) : root)
end

function pseudo_sqrt_inv_sqrt(M; cutoff::Real = defaulttol(M))
    @assert length(inds(M)) == 2
    cutoff >= zero(cutoff) || throw(ArgumentError("PSD pseudoinverse cutoff must be nonnegative"))
    M = parity_message_gauge(M)
    Q, D, Qdag = eigendecomp(M, inds(M)[1], inds(M)[2]; ishermitian = true)
    #After the per-sector gauge above, the environments are positive semidefinite in
    #exact arithmetic (a structural NSD fermionic block has already been sign-flipped,
    #not discarded). Eigensolvers can nevertheless return small negative modes, which
    #must be projected out rather than passed to sqrt, especially when cutoff == 0.
    #Keep the result in the eigenvalue scalar type for complex CUDA storage.
    D_sqrt = map_diag(x -> _psd_root_eigenvalue(x, cutoff, false), D)
    D_inv_sqrt = map_diag(x -> _psd_root_eigenvalue(x, cutoff, true), D)
    M_sqrt = Q * D_sqrt * Qdag
    M_inv_sqrt = Q * D_inv_sqrt * Qdag
    return M_sqrt, M_inv_sqrt
end

#TODO: Make this work for non-hermitian A
function eigendecomp(A, linds, rinds; ishermitian = false, kwargs...)
    @assert ishermitian
    D, U = eigen(A, linds, rinds; ishermitian, kwargs...)
    ul, ur = noncommonind(D, U), commonind(D, U)
    Ul = replaceinds(U, vcat(rinds, ur), vcat(linds, ul))
    return Ul, D, dag(U)
end

# Adapt `t` to the storage datatype (eltype + device) of `ref`.
adapt_like(ref, t) = adapt(datatype(ref))(t)

#Collect tensors narrowed to their backend type with the type PARAMETERS stripped:
#narrowing to a fully concrete eltype (what a plain `identity.(...)` does on
#rank-uniform networks, e.g. a 2×2 grid or a periodic lattice) freezes the tensor
#Dictionary's rank and rejects later rank-changing setindex! (sampling projections,
#charge legs). Same convention as `tensortype` and the message vectors.
narrow_tensors(ts) = collect(unspecify_type_parameters(mapreduce(typeof, typejoin, ts)), ts)

#Dangling "Charge" legs (charged graded states and projectors) pair bra-ket directly,
#like site legs with no operator: unprime them on a bra built as dag(prime(ket)).
function unprime_charge_legs(bra, ket)
    cinds = filter(i -> occursin("Charge", tags(i)), collect(inds(ket)))
    return isempty(cinds) ? bra : replaceinds(bra, prime.(cinds), cinds)
end

function identity_tensor(eltype, row_inds::Vector, col_inds::Vector)
    c_row, c_col = combiner(row_inds),combiner(col_inds)
    t = delta(eltype, combinedind(c_row), combinedind(c_col))
    return (t * c_row)*c_col
end

#Function for checking the correct algorithm is being used for the given cache type and functionality
function algorithm_check(tns::Union{AbstractBeliefPropagationCache, TensorNetworkState}, f::String, alg)
    if alg == "bp"
        if !((tns isa BeliefPropagationCache) || (tns isa TensorNetworkState))
            return error("Expected BeliefPropagationCache or TensorNetworkState for 'bp' algorithm, got $(typeof(tns))")
        end
        if f ∈ ["sample_certified"]
            return error("Certified sampling needs an estimate of p(x)/q(x), which the 'bp' sampler does not produce. Use alg = 'boundarymps'.")
        end
    elseif alg == "loopcorrections"
        if !((tns isa BeliefPropagationCache) || (tns isa TensorNetworkState))
            return error("Expected BeliefPropagationCache or TensorNetworkState for 'loop correction' algorithm, got $(typeof(tns))")
        end

        if f ∈ ["normalize", "expect", "sample", "sample_certified", "truncate", "rdm"]
            return error("Loop correction-based contraction not supported for this functionality yet")
        end
    elseif alg == "boundarymps"
        if !((tns isa BoundaryMPSCache) || (tns isa TensorNetworkState))
            return error("Expected BoundaryMPSCache or TensorNetworkState for 'boundarymps' algorithm, got $(typeof(tns))")
        end
        if f ∈ ["normalize"]
            return error("boundarymps contraction not supported for this functionality yet")
        end
    elseif alg == "exact"
        if f ∈ ["normalize", "sample", "sample_certified", "truncate"]
            return error("exact contraction not supported for this functionality yet")
        end
    else
        return error("Unrecognized algorithm specified. Must be one of 'exact', 'bp', 'loopcorrections', or 'boundarymps'")
    end
    return nothing
end

#Steiner region for a set of vertices and its incoming BP messages (shared by expect
#and reduced_density_matrix).
function bp_region(cache::BeliefPropagationCache, vs::Vector)
    steiner_vs = length(vs) == 1 ? vs : collect(vertices(steiner_tree(network(cache), vs)))
    return steiner_vs, incoming_messages(cache, steiner_vs)
end

#Contraction of a region closure [norm factors; incoming messages] with an optimal
#sequence. TODO: for large regions (≳100 tensors) the "optimal" search may become the
#bottleneck and a custom sequence is warranted.
function contract_bp_region(cache::BeliefPropagationCache, steiner_vs, incoming_ms; op_strings, joint_op = nothing)
    tensors = norm_factors(network(cache), steiner_vs; op_strings, joint_op)
    append!(tensors, incoming_ms)
    seq = contraction_sequence(tensors; alg = "optimal")
    return contract(tensors; sequence = seq)
end

#Default sequence-search settings for exact contractions (shared by expect, norm_sqr,
#inner, rdm and contract).
default_contraction_sequence_kwargs() = (; alg = "omeinsum", optimizer = GreedyMethod())

default_alg(bp_cache::BeliefPropagationCache) = "bp"
default_alg(bmps_cache::BoundaryMPSCache) = "boundarymps"
default_alg(any) = error("You must specify a contraction algorithm. Currently supported: exact, bp and boundarymps.")

# Build the cache `alg` needs over `network` and run it to convergence. Shared by the state-level
# entry points of `expect`, `reduced_density_matrix`, `norm_sqr`, `inner` and `normalize`, which all
# wrap a network in a cache, converge it, then hand off to the corresponding cache-level method.
# `cache_update_kwargs` is deliberately required: each entry point documents its own default.
function converged_cache(
        ::Union{Algorithm"bp", Algorithm"loopcorrections"}, network;
        cache_update_kwargs,
    )
    return update(BeliefPropagationCache(network); cache_update_kwargs...)
end

function converged_cache(
        ::Algorithm"boundarymps", network;
        mps_bond_dimension::Integer,
        partition_by = "row",
        gauge_state = false,
        cache_update_kwargs,
    )
    # `update` applies the `maxiter` default itself via `set_default_kwargs`, so none is added here
    cache = BoundaryMPSCache(network, mps_bond_dimension; partition_by, gauge_state)
    return update(cache; cache_update_kwargs...)
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
