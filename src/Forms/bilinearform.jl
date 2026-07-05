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

#Constructor, bra is taken to be in the vector space of ket so the dual is taken
function BilinearForm(ket::TensorNetworkState, bra::TensorNetworkState)
    @assert graph(ket) == graph(bra)
    sinds = siteinds(ket)
    verts = collect(vertices(ket))
    bra = TensorNetworkState(Dictionary(verts, [bra_tensor(bra, v) for v in verts]))
    operator_tensors = [reduce(*, ITensor[delta(scalartype(ket), dag(sind), prime(sind)) for sind in sinds[v]]) for v in verts]
    operator = TensorNetworkState(Dictionary(verts, operator_tensors))
    return BilinearForm(ket, operator, bra)
end
