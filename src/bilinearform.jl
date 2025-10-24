struct BilinearForm{V} <: AbstractITensorNetwork{V}
    ket::TensorNetworkState{V}
    operator::TensorNetworkState{V}
    bra::TensorNetworkState{V}
end

ket(blf::BilinearForm) = blf.ket
operator(blf::BilinearForm) = blf.operator
bra(blf::BilinearForm) = blf.bra

Base.copy(blf::BilinearForm) = BilinearForm(copy(blf.ket), copy(blf.operator), copy(blf.bra))

#Forward onto the ket
for f in [
    :(ITensorNetworks.underlying_graph),
    :(ITensorNetworks.data_graph_type),
    :(ITensorNetworks.data_graph),
    :(ITensors.datatype),
    :(ITensors.NDTensors.scalartype),
    :(NamedGraphs.edgeinduced_subgraphs_no_leaves)
]
@eval begin
    function $f(blf::BilinearForm, args...; kwargs...)
        return $f(ket(blf), args...; kwargs...)
    end
end
end

#Constructor, bra is taken to be in the vector space of ket so the dual is taken
function BilinearForm(ket::TensorNetworkState, bra::TensorNetworkState)
    @assert underlying_graph(ket) == underlying_graph(bra)
    @assert siteinds(ket) == siteinds(bra)
    bra = prime(dag(bra))
    sinds = siteinds(ket)
    verts = collect(vertices(ket))
    operator_tensors = [reduce(prod, ITensor[delta(sind, prime(dag(sind))) for sind in sinds[v]]) for v in verts]
    operator = TensorNetworkState(verts, operator_tensors)
    return BilinearForm(ket, operator, bra)
end

function default_message(blf::BilinearForm, edge::AbstractEdge)
    linds = ITensorNetworks.linkinds(blf, edge)
    return adapt(datatype(blf))(denseblocks(delta(linds)))
end

function ITensorNetworks.linkinds(blf::BilinearForm, edge::NamedEdge)
    return Index[linkinds(ket(blf), edge); linkinds(operator(blf), edge); linkinds(bra(blf), edge)]
end

function bp_factors(blf::BilinearForm, verts::Vector)
    factors = ITensor[]
    for v in verts
        append!(factors, ITensor[ket(blf)[v], operator(blf)[v],bra(blf)[v]])
    end
    return factors
end

bp_factors(blf::BilinearForm, v) = bp_factors(blf, [v])