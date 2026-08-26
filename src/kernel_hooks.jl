#=
Backend kernel hooks: the one place where backend-specific fast paths attach to the
generic algorithms. Each generic entry point (`updated_message`, `expect`'s region
contraction, `vertex_scalar`, `simple_update`) calls its hook first; the generic fallback
methods (defined next to those entry points) return `nothing`, so removing this file — or
running networks whose structure the pattern checks reject — leaves the library on the
plain seam-verb path with identical results.

The kernels themselves (buffered contraction chains, fused conjugation, in-place
factorizations) live in the KTensors module; these methods only translate network-level
structure (site indices, environment lists) into kernel inputs.
=#

#Fused fast path for the double-layer BP message update (see KTensors.fused_norm_message).
#Falls through to the generic contraction path when the message structure doesn't match
#(e.g. boundary-MPS messages with link indices).
function norm_message_kernel(tns::TensorNetworkState, v, incoming_ms::Vector{<:KTensor}; normalize)
    ψ = tns[v]
    ψ isa KTensor || return nothing
    sinds = siteinds(tns, v)
    all(i -> i isa KIndex, sinds) || return nothing
    return KTensors.fused_norm_message(ψ, collect(KIndex, sinds), incoming_ms; normalize)
end

#Fused fast path for BP region scalars (expectation-value numerators/denominators and
#vertex scalars): one- and two-vertex regions close through the same fused kernel — a
#two-vertex region is "message from v1 with its operator inserted" followed by a full
#closure at v2. Larger Steiner regions and non-standard structures fall back.
function norm_scalar_kernel(tns::TensorNetworkState, vs::Vector, incoming_ms::Vector{<:KTensor}; op_strings::Function)
    1 <= length(vs) <= 2 || return nothing
    ψs, sindss, ops = KTensor[], Vector{KIndex}[], Union{Nothing, KTensor}[]
    for v in vs
        ψ = tns[v]
        ψ isa KTensor || return nothing
        sinds = siteinds(tns, v)
        all(i -> i isa KIndex, sinds) || return nothing
        str = op_strings(v)
        if str == "I"
            push!(ops, nothing)
        elseif str == "ρ" || length(sinds) != 1
            return nothing
        else
            push!(ops, adapt_like(ψ, op(str, only(sinds))))
        end
        push!(ψs, ψ)
        push!(sindss, collect(KIndex, sinds))
    end

    if length(vs) == 1
        c = KTensors.fused_norm_closure(ψs[1], sindss[1], incoming_ms; op = ops[1])
        (c === nothing || !isempty(inds(c))) && return nothing
        return scalar(c)
    end

    #Partition the region's incoming messages by which vertex tensor they attach to
    ms1, ms2 = KTensor[], KTensor[]
    for m in incoming_ms
        ket_legs = filter(i -> plev(i) == 0, inds(m))
        if all(i -> i ∈ inds(ψs[1]), ket_legs)
            push!(ms1, m)
        elseif all(i -> i ∈ inds(ψs[2]), ket_legs)
            push!(ms2, m)
        else
            return nothing
        end
    end
    T1 = KTensors.fused_norm_closure(ψs[1], sindss[1], ms1; op = ops[1])
    T1 === nothing && return nothing
    c = KTensors.fused_norm_closure(ψs[2], sindss[2], vcat(ms2, [T1]); op = ops[2])
    (c === nothing || !isempty(inds(c))) && return nothing
    return scalar(c)
end

#Fused fast path for the two-site gate (see KTensors.fused_two_site_gate). Falls back on
#unusual apply_kwargs, empty environments, or non-2-index environments.
function fused_simple_update(
        o::KTensor, ψ⃗::Vector{<:KTensor};
        envs, normalize_tensors = true, sqrt_cutoff = nothing, apply_kwargs...
    )
    length(ψ⃗) == 2 || return nothing
    isempty(envs) && return nothing
    all(env -> env isa KTensor && ndims(env) == 2, envs) || return nothing
    isempty(setdiff(keys(apply_kwargs), (:maxdim, :cutoff))) || return nothing

    sqrt_cutoff = isnothing(sqrt_cutoff) ? 10 * eps(real(scalartype(first(envs)))) : sqrt_cutoff
    envs_v1 = filter(env -> hascommoninds(env, ψ⃗[1]), envs)
    envs_v2 = filter(env -> hascommoninds(env, ψ⃗[2]), envs)
    ssi1 = pseudo_sqrt_inv_sqrt.(envs_v1; cutoff = sqrt_cutoff)
    ssi2 = pseudo_sqrt_inv_sqrt.(envs_v2; cutoff = sqrt_cutoff)
    s1 = collect(KIndex, commoninds(ψ⃗[1], o))
    s2 = collect(KIndex, commoninds(ψ⃗[2], o))

    t1, t2, s_values, err = KTensors.fused_two_site_gate(
        o, ψ⃗[1], ψ⃗[2],
        collect(KTensor, first.(ssi1)), collect(KTensor, last.(ssi1)),
        collect(KTensor, first.(ssi2)), collect(KTensor, last.(ssi2)),
        s1, s2; apply_kwargs...
    )
    updated_tensors = [t1, t2]

    if normalize_tensors
        s_values = normalize(s_values)
        for ψᵥ in updated_tensors
            rmul!(data(ψᵥ), inv(norm(ψᵥ)))
        end
    end
    return noprime.(updated_tensors), s_values, err
end

#Direct entry point for circuits already given as backend tensors
function apply_gates(circuit::Vector{<:KTensor}, ψ_bpc::BeliefPropagationCache; kwargs...)
    return _apply_gate_tensors(circuit, ψ_bpc; kwargs...)
end
