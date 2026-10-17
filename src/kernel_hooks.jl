#=
Backend kernel hooks, gathered in one file so the generic algorithms stay backend-agnostic
(ported from FixesV2's src/kernel_hooks.jl, without its graded-only constructors).

The one specialised path is the double-layer BP closure (message updates and single-vertex
region scalars): the seam hooks `fused_norm_message` / `fused_norm_scalar` evaluate it without
forming the ket⊗bra tensor, keeping live memory at ψ plus two buffers. A backend that has no
fused kernel, or a structure it does not recognise — boundary-MPS messages with MPS link legs,
ρ insertions, operators on multi-site-index vertices, >1-vertex regions, graded tensors —
returns `nothing`, and the caller takes the generic contraction with identical results.
=#

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
