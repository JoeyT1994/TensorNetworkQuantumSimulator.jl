function normalize_rdm(ρ::ITensor)
    dtype = datatype(ρ)
    tr_ρ = copy(ρ)
    for i in inds(ρ; plev = 0)
        tr_ρ *= adapt(dtype)(delta(i, prime(i)))
    end
    return ρ / tr_ρ[]
end

"""
    reduced_density_matrix(ψ::Union{TensorNetworkState, BeliefPropagationCache, BoundaryMPSCache}, verts; alg::Union{String, Nothing} = nothing, kwargs...)
    Compute the reduced density matrix on the vertices `verts` of the tensor network state `ψ`.

    # Arguments
    - `ψ::Union{TensorNetworkState, BeliefPropagationCache, BoundaryMPSCache}`: The tensor network state or its associated cache.
    - `verts`: The vertices over which to compute the reduced density matrix. Can be a single vertex, or a collection of vertices.
    - `alg::Union{String, Nothing}`: (Optional) The contraction algorithm to use. If not provided, defaults based on the type of `ψ`. Supported algorithms are "exact", "bp", and "boundarymps".
    # Keyword Arguments
    - `kwargs...`: Additional keyword arguments specific to the chosen algorithm.
    - `normalize::Bool = true`: Whether to normalize the reduced density matrix so that its trace is 1.
    # Returns
    - An `ITensor` representing the reduced density matrix on the specified vertices.
"""
function reduced_density_matrix(ψ::Union{TensorNetworkState, BeliefPropagationCache, BoundaryMPSCache}, verts; alg::Union{String, Nothing} = default_alg(ψ), kwargs...)
    algorithm_check(ψ, "rdm", alg)
    verts = collect_vertices(verts, graph(ψ))
    return reduced_density_matrix(Algorithm(alg), ψ, verts; kwargs...)
end

function reduced_density_matrix(
        alg::Algorithm"exact",
        ψ::TensorNetworkState,
        verts::Vector;
        contraction_sequence_kwargs = (; alg = "einexpr", optimizer = Greedy()),
        normalize = true
    )
    ITensors.disable_warn_order()
    op_string_f = v -> v ∈ verts ? "ρ" : "I"
    ρ_tensors = norm_factors(ψ, collect(vertices(ψ)); op_strings = op_string_f)
    seq = contraction_sequence(ρ_tensors; contraction_sequence_kwargs...)
    ρ = contract(ρ_tensors; sequence = seq)
    if normalize
        ρ = normalize_rdm(ρ)
    end
    return ρ
end


function reduced_density_matrix(
        alg::Algorithm"bp",
        cache::BeliefPropagationCache,
        vs::Vector;
        normalize = true
    )
    steiner_vs = length(vs) == 1 ? vs : collect(vertices(steiner_tree(network(cache), vs)))
    incoming_ms = incoming_messages(cache, steiner_vs)

    op_string_f = v -> v ∈ vs ? "ρ" : "I"

    #TODO: If there are a lot of tensors here, (more than 100 say), we need to think about defining a custom sequence as optimal may be too slow
    ρ_tensors = norm_factors(network(cache), steiner_vs; op_strings = op_string_f)
    append!(ρ_tensors, incoming_ms)
    seq = contraction_sequence(ρ_tensors; alg = "optimal", prune_tensors = true)
    ρ = contract(ρ_tensors; sequence = seq)

    if normalize
        ρ = normalize_rdm(ρ)
    end
    return ρ
end

function reduced_density_matrix(
        alg::Algorithm"boundarymps",
        cache::BoundaryMPSCache,
        vs::Vector;
        normalize = true,
        bmps_messages_up_to_date = false,
    )

    op_string_f = v -> v ∈ vs ? "ρ" : "I"
    ρ, _ = path_contract(cache, vs, op_string_f; bmps_messages_up_to_date)

    if normalize
        ρ = normalize_rdm(ρ)
    end
    return ρ
end

function reduced_density_matrix(
        alg::Algorithm"bp",
        ψ::TensorNetworkState,
        verts::Vector;
        cache_update_kwargs = default_bp_update_kwargs(ψ),
        kwargs...,
    )
    ψ_bpc = BeliefPropagationCache(ψ)
    ψ_bpc = update(ψ_bpc; cache_update_kwargs...)

    return reduced_density_matrix(alg, ψ_bpc, verts; kwargs...)
end

function reduced_density_matrix(
        alg::Algorithm"boundarymps",
        ψ::TensorNetworkState,
        verts::Vector;
        cache_update_kwargs = default_bmps_update_kwargs(ψ),
        mps_bond_dimension::Integer,
        partition_by::String = boundarymps_partitioning(verts),
        kwargs...,
    )
    ψ_bpc = BoundaryMPSCache(ψ, mps_bond_dimension; partition_by)
    ψ_bpc = update(ψ_bpc; cache_update_kwargs...)

    return reduced_density_matrix(alg, ψ_bpc, verts; kwargs...)
end

function boundarymps_partitioning(vs::Vector)
    allequal(first.(vs)) && return "row"
    allequal(last.(vs)) && return "col"
    error("Vertices must be aligned in either the same column or the same row to do BoundaryMPS.")
end

const rdm = reduced_density_matrix
