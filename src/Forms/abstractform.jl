# A Form wraps a `ket`, an `operator` and a `bra` tensor network (the bra may be stored
# or derived from the ket). Concrete subtypes must define `ket`, `operator`, `bra`, and the
# per-vertex / per-edge dual accessors `bra_tensor` and `bra_virtualinds`.
abstract type AbstractForm{V} <: AbstractTensorNetwork{V} end

#Forward onto the ket
for f in [
        :(graph),
        :(ITensors.datatype),
        :(ITensors.NDTensors.scalartype),
        :(NamedGraphs.leafless_edge_induced_subgraphs),
    ]
    @eval begin
        function $f(form::AbstractForm, args...; kwargs...)
            return $f(ket(form), args...; kwargs...)
        end
    end
end

function virtualinds(form::AbstractForm, edge::NamedEdge)
    return Index[virtualinds(ket(form), edge); virtualinds(operator(form), edge); bra_virtualinds(form, edge)]
end

function default_message(form::AbstractForm, edge::AbstractEdge)
    return adapt_like(form, denseblocks(delta(virtualinds(form, edge))))
end

function bp_factors(form::AbstractForm, verts::Vector)
    factors = ITensor[]
    for v in verts
        append!(factors, ITensor[ket(form)[v], operator(form)[v], bra_tensor(form, v)])
    end
    return factors
end

bp_factors(form::AbstractForm, v) = bp_factors(form, [v])
