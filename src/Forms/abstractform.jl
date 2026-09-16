# A Form wraps a `ket`, an `operator` and a `bra` tensor network (the bra may be stored
# or derived from the ket). Concrete subtypes must define `ket`, `operator`, and the
# per-vertex / per-edge dual accessors `bra_tensor` and `bra_virtualinds`. A whole-network
# `bra` is optional: `QuadraticForm` derives its bra lazily and does not provide one.
abstract type AbstractForm{V} <: AbstractTensorNetwork{V} end

ket(form::AbstractForm) = not_implemented()
operator(form::AbstractForm) = not_implemented()
bra_tensor(form::AbstractForm, v) = not_implemented()
bra_virtualinds(form::AbstractForm, edge) = not_implemented()

#Forward onto the ket
for f in [
        :(graph),
        :datatype,
        :(scalartype),
        :tensortype,
        :(NamedGraphs.leafless_edge_induced_subgraphs),
    ]
    @eval begin
        function $f(form::AbstractForm, args...; kwargs...)
            return $f(ket(form), args...; kwargs...)
        end
    end
end

function virtualinds(form::AbstractForm, edge::NamedEdge)
    return vcat(virtualinds(ket(form), edge), virtualinds(operator(form), edge), bra_virtualinds(form, edge))
end

function default_message(form::AbstractForm, edge::AbstractEdge)
    # Identity between the ket and bra bonds. An operator layer with its own virtual legs (the
    # generating operator's auxiliary index) starts in its first slot — the norm sector — rather
    # than on a three-leg diagonal: on a graded backend a lone auxiliary leg cannot be paired by
    # `delta` at all (fermionic sites, 2026-09-15), and the a > 0 slots are sourced by the vertex
    # factors during the iteration anyway.
    m = delta(vcat(virtualinds(ket(form), edge), bra_virtualinds(form, edge)))
    for a in virtualinds(operator(form), edge)
        m = m * TensorInterface.onehot(Float64, a => 1)
    end
    return adapt_like(form, m)
end

function bp_factors(form::AbstractForm, verts::Vector)
    factors = tensortype(ket(form))[]
    for v in verts
        append!(factors, [ket(form)[v], operator(form)[v], bra_tensor(form, v)])
    end
    return factors
end

bp_factors(form::AbstractForm, v) = bp_factors(form, [v])
