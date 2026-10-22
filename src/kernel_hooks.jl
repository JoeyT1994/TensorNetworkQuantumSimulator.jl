# Fused double-layer BP paths (message updates and single-vertex scalars) that avoid forming the
# ket⊗bra tensor. Both return `nothing` for anything the backend kernel does not handle
# (non-tensor messages, ρ insertions, multi-vertex regions, ...) and the caller falls back to the
# generic contraction.

function norm_message_kernel(tns::TensorNetworkState, v, incoming_ms::Vector; normalize)
    all(m -> m isa Tensor, incoming_ms) || return nothing
    return fused_norm_message(tns[v], siteinds(tns, v), incoming_ms; normalize)
end

function norm_scalar_kernel(tns::TensorNetworkState, vs::Vector, incoming_ms::Vector; op_strings::Function)
    all(m -> m isa Tensor, incoming_ms) || return nothing
    length(vs) == 1 || return nothing
    v = only(vs)
    sinds = siteinds(tns, v)
    name = op_strings(v)
    name isa String || return nothing
    ops = Tensor[]
    if name != "I"
        (name == "ρ" || length(sinds) != 1) && return nothing
        push!(ops, adapt_like(tns[v], op(name, only(sinds))))
    end
    return fused_norm_scalar(tns[v], sinds, incoming_ms, ops)
end
