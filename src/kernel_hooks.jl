# Fused double-layer BP paths (message updates and single-vertex scalars) that never form the
# ket⊗bra tensor. `nothing` (e.g. for ρ insertions or multi-vertex regions) means the caller takes
# the generic contraction.
norm_message_kernel(net, vertex, incoming_ms) = nothing
norm_scalar_kernel(net, vertices, incoming_ms; kwargs...) = nothing

# The kernels pool their scratch buffers in a workspace, which a caller (e.g. a BP update) may
# already have opened.
function norm_message_kernel(tns::TensorNetworkState, v, incoming_ms::Vector)
    return with_kernel_workspace(() -> fused_norm_message(tns[v], siteinds(tns, v), incoming_ms))
end

function norm_scalar_kernel(tns::TensorNetworkState, vs::Vector, incoming_ms::Vector; op_strings::Function)
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
    return with_kernel_workspace(() -> fused_norm_scalar(tns[v], sinds, incoming_ms, ops))
end
