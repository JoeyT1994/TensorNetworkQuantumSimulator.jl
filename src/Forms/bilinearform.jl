struct BilinearForm{V} <: AbstractForm{V}
    ket::TensorNetworkState{V}
    operator::TensorNetworkState{V}
    bra::TensorNetworkState{V}
end

ket(blf::BilinearForm) = blf.ket
operator(blf::BilinearForm) = blf.operator
bra(blf::BilinearForm) = blf.bra
bra_tensor(blf::BilinearForm, v) = bra(blf)[v]
bra_virtualinds(blf::BilinearForm, edge::NamedEdge) = virtualinds(bra(blf), edge)

Base.copy(blf::BilinearForm) = BilinearForm(copy(blf.ket), copy(blf.operator), copy(blf.bra))

#Constructor, bra is taken to be in the vector space of ket so the dual is taken.
#The identity operator carries the PROMOTED scalar type: a real ket with a complex bra
#(or vice versa) must not silently truncate either side.
function BilinearForm(ket::TensorNetworkState, bra::TensorNetworkState)
    graph(ket) == graph(bra) || error("BilinearForm: ket and bra must share the same graph")
    elt = promote_type(scalartype(ket), scalartype(bra))
    dtype = Base.typename(datatype(ket)).wrapper{elt, 1}
    bra = map_tensors(t -> dag(prime(t)), bra)
    sinds = siteinds(ket)
    verts = collect(vertices(ket))
    operator_tensors = [adapt(dtype)(reduce(*, [delta(sind, prime(dag(sind))) for sind in sinds[v]])) for v in verts]
    operator = TensorNetworkState(Dictionary(verts, operator_tensors))
    return BilinearForm(ket, operator, bra)
end

#scalar/storage types must span both stored networks (the generic AbstractForm methods
#forward to the ket only, which under-reports for an independently-typed bra)
scalartype(blf::BilinearForm) = promote_type(scalartype(blf.ket), scalartype(blf.bra))
