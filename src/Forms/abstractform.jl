# A Form wraps a `ket` and a `bra` tensor network. Concrete subtypes define
# `ITNN.ketnetwork`, `ITNN.branetwork` and `ITNN.factor_tensors`.
abstract type AbstractForm{T, V} <: AbstractTensorNetwork{T, V} end

#Forward onto the ket
for f in [
        :(Graphs.vertices),
        :(Graphs.edges),
        :(NamedGraphs.encoded_vertex),
        :(NamedGraphs.decoded_vertex),
        :(NamedGraphs.encoded_graph),
        :(DataGraphs.is_vertex_assigned),
    ]
    @eval begin
        function $f(form::AbstractForm, args...; kwargs...)
            return $f(ITNN.ketnetwork(form), args...; kwargs...)
        end
    end
end

for f in [
        :(datatype),
        :(VectorInterface.scalartype),
    ]
    @eval begin
        function $f(form::Union{AbstractForm, ITNN.AbstractBilinearFormNetwork}, args...; kwargs...)
            return $f(ITNN.ketnetwork(form), args...; kwargs...)
        end
    end
end

function virtualinds(form::Union{AbstractForm, ITNN.AbstractBilinearFormNetwork}, edge::NamedEdge)
    return Index[ITNN.linkinds(ITNN.ketnetwork(form), edge); ITNN.linkinds(ITNN.branetwork(form), edge)]
end
